"""The dev / held-out split, and the guarantee that it cannot quietly change.

The brief forbids tuning on the held-out split. That prohibition is unenforceable
unless the split is fixed before any number is looked at and can be shown later to be
the same split, so most of these tests are about immutability and reproducibility
rather than about balance.

The last test is the important one: it recomputes the split from the real corpus and
compares it to the file in version control. If anyone changes the seed, the fraction,
the stratification or the dealing rule, that test fails and says that every number
already reported has become incomparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentpath_eval.corpus_split import (
    DEV,
    HELD_OUT,
    HELD_OUT_FRACTION,
    SPLIT_FORMAT_VERSION,
    SPLIT_SEED,
    SplitQuestion,
    load_split,
    make_split,
    split_path,
    write_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MRAMG_BUILD = REPO_ROOT / "data" / "external" / "mramg" / "build"

needs_corpus = pytest.mark.skipif(
    not (MRAMG_BUILD / "questions.jsonl").is_file(),
    reason="the MRAMG bundle is not exported; run: make corpus-fetch && make corpus-export",
)


@dataclass(frozen=True)
class FakeQuestion:
    """Only the two attributes the split is allowed to see."""

    question_id: str
    ground_truth_image_ids: tuple[str, ...] = ()
    difficulty: str | None = None


def _questions(counts: dict[int, int]) -> list[FakeQuestion]:
    """``{image_count: how_many}`` -> questions, in a deliberately unsorted order."""
    made: list[FakeQuestion] = []
    for image_count, how_many in counts.items():
        for index in range(how_many):
            made.append(
                FakeQuestion(
                    question_id=f"Q{image_count}_{index:03d}",
                    ground_truth_image_ids=tuple(f"img{n}" for n in range(image_count)),
                )
            )
    # Reversed, so any test that passes would also have passed on sorted input only by
    # accident.
    return list(reversed(made))


# --------------------------------------------------------------------------- #
# Every question on exactly one side
# --------------------------------------------------------------------------- #


def test_every_question_lands_on_exactly_one_side() -> None:
    questions = _questions({0: 7, 1: 20, 2: 13})

    split = make_split(questions, corpus_id="demo")

    assert split.total == len(questions)
    assert not set(split.dev) & set(split.held_out)
    assert set(split.dev) | set(split.held_out) == {question.question_id for question in questions}


def test_an_empty_corpus_produces_an_empty_split() -> None:
    split = make_split([], corpus_id="demo")

    assert split.total == 0
    assert split.dev == ()
    assert split.strata == {}


def test_a_single_question_is_not_lost() -> None:
    split = make_split([FakeQuestion("only")], corpus_id="demo")

    assert split.total == 1
    assert split.side_of("only") in {DEV, HELD_OUT}


def test_a_question_that_is_in_neither_side_reports_none() -> None:
    split = make_split([FakeQuestion("a"), FakeQuestion("b")], corpus_id="demo")

    assert split.side_of("never-existed") is None


# --------------------------------------------------------------------------- #
# Reproducibility, which is the whole point
# --------------------------------------------------------------------------- #


def test_the_same_questions_give_the_same_split_every_time() -> None:
    questions = _questions({0: 5, 1: 9, 2: 14})

    first = make_split(questions, corpus_id="demo")
    second = make_split(questions, corpus_id="demo")

    assert first.dev == second.dev
    assert first.held_out == second.held_out


def test_the_input_order_does_not_change_the_split() -> None:
    """Re-exporting the bundle in a different order must not resplit the corpus.

    Without the sort inside ``make_split`` this is exactly how every previously
    reported number would silently stop being comparable.
    """
    questions = _questions({0: 5, 1: 9, 2: 14})

    ordered = make_split(sorted(questions, key=lambda q: q.question_id), corpus_id="demo")
    shuffled = make_split(list(reversed(questions)), corpus_id="demo")

    assert ordered.dev == shuffled.dev
    assert ordered.held_out == shuffled.held_out


def test_a_different_seed_gives_a_different_split() -> None:
    """Otherwise the seed would be decoration rather than the thing that decided."""
    questions = _questions({1: 40, 2: 40})

    first = make_split(questions, corpus_id="demo", seed=1)
    second = make_split(questions, corpus_id="demo", seed=2)

    assert first.dev != second.dev


def test_a_different_corpus_id_gives_a_different_split() -> None:
    """The corpus id is in the per-stratum seed, so two corpora with the same question
    ids do not receive the same assignment."""
    questions = _questions({1: 40})

    assert make_split(questions, corpus_id="a").dev != make_split(questions, corpus_id="b").dev


def test_adding_a_new_stratum_does_not_reshuffle_the_existing_ones() -> None:
    """The per-stratum seed exists for this: a corpus that gains six-image questions
    must not resplit its one-image questions."""
    base = _questions({1: 20, 2: 20})
    extended = base + _questions({6: 5})

    first = make_split(base, corpus_id="demo")
    second = make_split(extended, corpus_id="demo")

    original_ids = {question.question_id for question in base}
    assert [item for item in second.dev if item in original_ids] == list(first.dev)


# --------------------------------------------------------------------------- #
# Stratification
# --------------------------------------------------------------------------- #


def test_each_stratum_is_balanced_to_within_one_question() -> None:
    """A stratum of nine sliced at a fraction would give its spare to the same side
    every time, and with seven strata that bias compounds."""
    split = make_split(_questions({0: 9, 1: 21, 2: 34, 6: 3}), corpus_id="demo")

    for stratum, (dev_count, held_count) in split.strata.items():
        assert abs(dev_count - held_count) <= 1, stratum


def test_the_stratum_is_the_expected_image_count_when_there_is_no_difficulty_label() -> None:
    """MRAMG-Manual labels no difficulty, so this is the axis available."""
    projected = SplitQuestion.of(FakeQuestion("q", ground_truth_image_ids=("a", "b", "c")))

    assert projected.stratum == "images=3"


def test_a_difficulty_label_is_preferred_when_the_corpus_has_one() -> None:
    """A corpus that does label difficulty gets the better axis without a code change."""
    projected = SplitQuestion.of(FakeQuestion("q", ground_truth_image_ids=("a",), difficulty="hard"))

    assert projected.stratum == "difficulty=hard"


def test_a_question_expecting_no_image_is_its_own_stratum() -> None:
    """40 of MRAMG-Manual's 390 expect none, and they are the only questions that can
    show whether image selection knows how to abstain."""
    split = make_split(_questions({0: 8, 1: 8}), corpus_id="demo")

    assert "images=0" in split.strata
    assert split.strata["images=0"] == (4, 4)


def test_the_split_is_projected_and_cannot_read_the_ground_truth_answer() -> None:
    """The split must not be able to see the answers it is splitting."""
    fields = set(SplitQuestion.__dataclass_fields__)

    assert fields == {"question_id", "stratum"}


# --------------------------------------------------------------------------- #
# Selecting from a side
# --------------------------------------------------------------------------- #


def test_a_sample_of_a_side_is_a_stable_prefix() -> None:
    """The generation evaluation runs on 50 held-out questions, and it has to be the
    same 50 on a re-run without introducing a second seed."""
    split = make_split(_questions({1: 40, 2: 40}), corpus_id="demo")

    first = split.sample(HELD_OUT, 10)
    second = split.sample(HELD_OUT, 10)

    assert first == second
    assert len(first) == 10
    assert list(first) == list(split.held_out[:10])


def test_a_sample_larger_than_the_side_returns_the_whole_side() -> None:
    split = make_split(_questions({1: 4}), corpus_id="demo")

    assert len(split.sample(HELD_OUT, 500)) == len(split.held_out)


def test_questions_for_a_side_come_back_in_the_splits_order() -> None:
    """The split's order, not the corpus's, so "the first 50" means one thing."""
    questions = _questions({1: 10, 2: 10})
    split = make_split(questions, corpus_id="demo")

    selected = split.questions_for(HELD_OUT, questions)

    assert [question.question_id for question in selected] == list(split.held_out)


def test_questions_for_a_side_skips_ids_the_corpus_no_longer_has() -> None:
    """A split outliving a question it names is a real possibility; it must not raise."""
    questions = _questions({1: 10})
    split = make_split(questions, corpus_id="demo")

    selected = split.questions_for(HELD_OUT, questions[:2])

    assert len(selected) <= 2


# --------------------------------------------------------------------------- #
# The file
# --------------------------------------------------------------------------- #


def test_the_written_split_round_trips(tmp_path: Path) -> None:
    split = make_split(_questions({0: 5, 1: 11}), corpus_id="demo")
    path = write_split(split, tmp_path / "demo.json")

    reloaded = load_split(path)

    assert reloaded.dev == split.dev
    assert reloaded.held_out == split.held_out
    assert reloaded.seed == split.seed
    assert reloaded.strata == split.strata


def test_the_file_records_the_seed_and_the_fraction(tmp_path: Path) -> None:
    """A table has to be able to name the split that produced it."""
    split = make_split(_questions({1: 6}), corpus_id="demo")
    payload = json.loads(write_split(split, tmp_path / "demo.json").read_text(encoding="utf-8"))

    assert payload["seed"] == SPLIT_SEED
    assert payload["held_out_fraction"] == HELD_OUT_FRACTION
    assert payload["format_version"] == SPLIT_FORMAT_VERSION
    assert payload["counts"]["total"] == 6


def test_a_split_file_from_a_future_format_is_refused(tmp_path: Path) -> None:
    path = write_split(make_split(_questions({1: 4}), corpus_id="demo"), tmp_path / "demo.json")
    path.write_text(path.read_text().replace('"format_version": 1', '"format_version": 99'))

    with pytest.raises(ValueError, match="format_version"):
        load_split(path)


def test_the_split_file_is_sorted_so_a_diff_is_readable(tmp_path: Path) -> None:
    split = make_split(_questions({0: 3, 6: 3, 2: 3}), corpus_id="demo")
    payload = json.loads(write_split(split, tmp_path / "demo.json").read_text(encoding="utf-8"))

    assert list(payload["strata"]) == sorted(payload["strata"])


# --------------------------------------------------------------------------- #
# The committed split for the real corpus
# --------------------------------------------------------------------------- #


def test_the_committed_split_exists_and_is_the_recorded_shape() -> None:
    """Committed, unlike every other artefact here: a split nobody can check cannot
    support the claim that nothing was tuned on it."""
    path = split_path("mramg-manual")

    assert path.is_file(), "run: make corpus-split"
    split = load_split(path)
    assert split.corpus_id == "mramg-manual"
    assert split.seed == SPLIT_SEED
    assert split.total == 390
    assert len(split.held_out) == 194
    assert len(split.dev) == 196


def test_the_committed_split_has_no_overlap_and_no_duplicates() -> None:
    split = load_split(split_path("mramg-manual"))

    assert not set(split.dev) & set(split.held_out)
    assert len(set(split.dev)) == len(split.dev)
    assert len(set(split.held_out)) == len(split.held_out)


def test_the_committed_split_holds_enough_for_the_generation_evaluation() -> None:
    """The brief asks for 50 held-out questions for answer generation."""
    split = load_split(split_path("mramg-manual"))

    assert len(split.sample(HELD_OUT, 50)) == 50


def test_every_stratum_of_the_committed_split_is_balanced() -> None:
    split = load_split(split_path("mramg-manual"))

    assert set(split.strata) == {f"images={n}" for n in range(7)}
    for stratum, (dev_count, held_count) in split.strata.items():
        assert abs(dev_count - held_count) <= 1, stratum


@needs_corpus
def test_recomputing_the_split_reproduces_the_committed_file_exactly() -> None:
    """The guarantee, asserted against the real corpus.

    If this fails, the seed, the fraction, the stratification or the dealing rule has
    changed, and every number already reported against the old split has become
    incomparable. Fix the code or accept a new split deliberately - do not adjust this
    test to make it pass.
    """
    from agentpath.corpus import CorpusCatalogue

    corpus = CorpusCatalogue.load().corpus("mramg-manual")
    assert corpus is not None

    recomputed = make_split(corpus.questions, corpus_id="mramg-manual")
    committed = load_split(split_path("mramg-manual"))

    assert recomputed.dev == committed.dev
    assert recomputed.held_out == committed.held_out
    assert recomputed.strata == committed.strata


@needs_corpus
def test_the_committed_split_names_only_questions_the_corpus_has() -> None:
    from agentpath.corpus import CorpusCatalogue

    corpus = CorpusCatalogue.load().corpus("mramg-manual")
    assert corpus is not None
    known = {question.question_id for question in corpus.questions}

    split = load_split(split_path("mramg-manual"))

    assert set(split.dev) | set(split.held_out) == known
