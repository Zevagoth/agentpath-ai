"""A fixed dev / held-out split of an external corpus's questions.

Why this exists at all: the brief forbids tuning on the held-out split, and that
prohibition is unenforceable unless the split is decided once, before any number is
looked at, and can be shown to be the same split later. So this module does three
things - compute the split deterministically, write it down, and let a test prove the
computation still reproduces what was written.

Three properties, each for a reason.

**Deterministic from a recorded seed.** The seed is a module constant and travels into
the split file, so "which split produced this table" is answerable from the table. The
shuffle is seeded per stratum from that one seed.

**Independent of input order.** Question ids are sorted before anything random happens.
Without that, re-exporting the bundle in a different order would silently produce a
different split, and every previously reported number would quietly stop being
comparable.

**Stratified on how many images the answer needs.** MRAMG-Manual ships no difficulty
label - ``difficulty`` is ``None`` for all 390 questions - so the available stratum is
``expected_image_count``, which is the axis that matters most here: 40 of the 390
questions expect no image at all, and a split that put most of those on one side would
make the image metrics of the two halves incomparable. The real distribution is
``{0: 40, 1: 92, 2: 126, 3: 74, 4: 37, 5: 12, 6: 9}``, so the tail strata are small
enough that an unstratified shuffle could plausibly put all nine six-image questions in
one half.

Half and half, rather than the usual 80/20. The held-out side is not a sanity check
here, it is the evidence the report rests on: it carries the retrieval table *and* the
50-question generation evaluation, and it has to be large enough that a difference
between methods is not noise. 195 questions on the dev side is still far more than
enough for any threshold decision.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

#: Recorded so a table can name the split that produced it. Changing this
#: invalidates every previously reported number, which is why it is a constant in
#: version control and not a command-line flag.
SPLIT_SEED = 20260919

#: Half. See the module docstring.
HELD_OUT_FRACTION = 0.5

#: Shape of the split file. Bumped if a reader would need to notice a change.
SPLIT_FORMAT_VERSION = 1

DEV = "dev"
HELD_OUT = "held_out"


@runtime_checkable
class QuestionLike(Protocol):
    """The three fields the split is allowed to see.

    A structural type rather than an import: the same split has to work for the API's
    ``CorpusQuestionRecord`` and the ingestion side's ``CorpusQuestion``, which are
    deliberately separate types, and depending on either would tie the evaluation to one
    of them. Naming the fields here also states the limit - the authored answer is not
    among them, so the split cannot be influenced by what it is splitting.
    """

    question_id: str
    ground_truth_image_ids: tuple[str, ...]
    difficulty: str | None


@dataclass(frozen=True, slots=True)
class SplitQuestion:
    """What the split needs to know about one question.

    A projection rather than the question itself: the split must not be able to read
    the ground-truth answer, so it is not given it.
    """

    question_id: str
    #: The stratum. For MRAMG-Manual, how many images the authored answer contains.
    stratum: str

    @classmethod
    def of(cls, question: QuestionLike) -> SplitQuestion:
        question_id = str(question.question_id)
        difficulty = question.difficulty
        if difficulty:
            # A corpus that *does* label difficulty gets stratified on the label, which
            # is the better axis when it exists.
            return cls(question_id=question_id, stratum=f"difficulty={difficulty}")
        images = question.ground_truth_image_ids or ()
        return cls(question_id=question_id, stratum=f"images={len(images)}")


@dataclass(frozen=True, slots=True)
class CorpusSplit:
    """Which questions are for tuning and which are evidence."""

    corpus_id: str
    seed: int
    held_out_fraction: float
    dev: tuple[str, ...]
    held_out: tuple[str, ...]
    #: Stratum -> (dev count, held-out count). Written out so the balance can be read
    #: rather than trusted.
    strata: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.dev) + len(self.held_out)

    def side_of(self, question_id: str) -> str | None:
        """Which side a question is on, or ``None`` when it is in neither."""
        if question_id in set(self.dev):
            return DEV
        if question_id in set(self.held_out):
            return HELD_OUT
        return None

    def questions_for(self, side: str, questions: Iterable[QuestionLike]) -> tuple[QuestionLike, ...]:
        """Filter *questions* to one side, in the split's own recorded order.

        The split's order, not the corpus's, so a run over "the first 50 held-out
        questions" means the same 50 in the same sequence every time.

        A question the split names but the corpus no longer has is skipped rather than
        raising: a split outlives the bundle it was computed from, and a re-export that
        drops a question should degrade the run, not abort it.
        """
        wanted = self.dev if side == DEV else self.held_out
        by_id = {str(question.question_id): question for question in questions}
        return tuple(by_id[question_id] for question_id in wanted if question_id in by_id)

    def sample(self, side: str, count: int) -> tuple[str, ...]:
        """The first *count* questions of one side.

        The first, not a fresh random draw: the side is already in a
        seed-determined order, so taking a prefix is reproducible without introducing a
        second seed whose value would also have to be recorded and defended.
        """
        wanted = self.dev if side == DEV else self.held_out
        return wanted[:count]

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": SPLIT_FORMAT_VERSION,
            "corpus_id": self.corpus_id,
            "seed": self.seed,
            "held_out_fraction": self.held_out_fraction,
            "counts": {"total": self.total, DEV: len(self.dev), HELD_OUT: len(self.held_out)},
            "strata": {
                name: {DEV: dev_count, HELD_OUT: held_count}
                for name, (dev_count, held_count) in sorted(self.strata.items())
            },
            DEV: list(self.dev),
            HELD_OUT: list(self.held_out),
        }


def make_split(
    questions: Sequence[QuestionLike],
    *,
    corpus_id: str,
    seed: int = SPLIT_SEED,
    held_out_fraction: float = HELD_OUT_FRACTION,
) -> CorpusSplit:
    """Assign every question to exactly one side, reproducibly.

    Within each stratum the questions are sorted, shuffled with a seed derived from the
    stratum's name, and then dealt one at a time by accumulating
    ``held_out_fraction`` and handing a question over each time the accumulator passes
    one. That keeps every stratum balanced to within a single question - measured on
    MRAMG-Manual: 20/20, 46/46, 63/63, 37/37, 19/18, 6/6, 5/4 - which slicing each
    stratum at a fraction would not, since it rounds the same way every time and would
    give every odd stratum's spare question to the same side.

    The deal starts on the dev side, so a stratum with an odd count leaves dev one
    larger (196 against 194 overall). Neither side is preferable for the spare question;
    what matters is that the rule is fixed and written down.
    """
    projected = [SplitQuestion.of(question) for question in questions]
    grouped: dict[str, list[str]] = {}
    for item in projected:
        grouped.setdefault(item.stratum, []).append(item.question_id)

    dev: list[str] = []
    held_out: list[str] = []
    strata: dict[str, tuple[int, int]] = {}

    for stratum in sorted(grouped):
        # Sorted first: the result must not depend on the order the corpus happened to
        # list its questions in.
        members = sorted(grouped[stratum])
        # A per-stratum seed, derived from the recorded one, so adding a stratum cannot
        # reshuffle the strata that already existed.
        rng = random.Random(f"{seed}:{corpus_id}:{stratum}")
        rng.shuffle(members)

        stratum_dev: list[str] = []
        stratum_held: list[str] = []
        held_target = held_out_fraction
        credit = 0.0
        for question_id in members:
            credit += held_target
            if credit >= 1.0:
                credit -= 1.0
                stratum_held.append(question_id)
            else:
                stratum_dev.append(question_id)

        dev.extend(stratum_dev)
        held_out.extend(stratum_held)
        strata[stratum] = (len(stratum_dev), len(stratum_held))

    return CorpusSplit(
        corpus_id=corpus_id,
        seed=seed,
        held_out_fraction=held_out_fraction,
        dev=tuple(dev),
        held_out=tuple(held_out),
        strata=strata,
    )


def write_split(split: CorpusSplit, path: Path) -> Path:
    """Write the split where it can be read by a human and diffed by git.

    Committed, unlike every other artefact this project writes. A split that lives
    only in a ``.gitignore``d directory is a split nobody can check, and "we did not
    tune on the held-out set" is a claim that needs checking.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_split(path: Path) -> CorpusSplit:
    """Read a split file, refusing a shape this code does not understand."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("format_version")
    if version != SPLIT_FORMAT_VERSION:
        raise ValueError(f"split file {path} has format_version {version!r}, expected {SPLIT_FORMAT_VERSION}")
    strata = {name: (int(counts[DEV]), int(counts[HELD_OUT])) for name, counts in (payload.get("strata") or {}).items()}
    return CorpusSplit(
        corpus_id=str(payload["corpus_id"]),
        seed=int(payload["seed"]),
        held_out_fraction=float(payload["held_out_fraction"]),
        dev=tuple(str(item) for item in payload[DEV]),
        held_out=tuple(str(item) for item in payload[HELD_OUT]),
        strata=strata,
    )


def split_path(corpus_id: str, root: Path | None = None) -> Path:
    """Where a corpus's committed split lives."""
    base = root if root is not None else Path(__file__).resolve().parents[1] / "splits"
    return base / f"{corpus_id}.json"


__all__ = [
    "DEV",
    "HELD_OUT",
    "HELD_OUT_FRACTION",
    "SPLIT_FORMAT_VERSION",
    "SPLIT_SEED",
    "CorpusSplit",
    "QuestionLike",
    "SplitQuestion",
    "load_split",
    "make_split",
    "split_path",
    "write_split",
]
