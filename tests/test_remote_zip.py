"""The range-request zip reader, against a real zip served by a fake transport.

No network. A zip is built in memory, handed to `httpx.MockTransport`, and read
back through the same code path that reads MRAMG's 1.53 GiB archive - so the
central-directory parsing, the range coalescing and the CRC verification are all
exercised against bytes a real zip tool produced.
"""

from __future__ import annotations

import io
import zipfile

import httpx
import pytest

from agentpath_ingestion.corpora.remote_zip import (
    DEFAULT_GAP_TOLERANCE,
    RemoteZip,
    RemoteZipError,
)

URL = "https://example.test/IMAGE.zip"


def build_zip(entries: dict[str, bytes], *, compress: bool = True) -> bytes:
    buffer = io.BytesIO()
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", method) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def serve(archive: bytes, *, honour_range: bool = True, calls: list[str] | None = None) -> httpx.Client:
    """A transport that behaves like the Hugging Face CDN."""

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(f"{request.method} {request.headers.get('range', '-')}")
        if request.method == "HEAD":
            return httpx.Response(200, headers={"x-linked-size": str(len(archive))})
        header = request.headers.get("range")
        if not header or not honour_range:
            return httpx.Response(200, content=archive)
        span = header.removeprefix("bytes=")
        start_text, _, end_text = span.partition("-")
        start = int(start_text)
        end = min(int(end_text), len(archive) - 1)
        return httpx.Response(
            206,
            content=archive[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(archive)}"},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- #
# Reading the index
# --------------------------------------------------------------------------- #


def incompressible(seed: int, size: int) -> bytes:
    """Bytes deflate cannot shrink, so a test archive is really as big as it looks.

    Needed because the reader searches the last 64 KiB for the index: an archive
    smaller than that is read in full by the search itself, and "it did not
    download the payloads" becomes untestable.
    """
    state = seed & 0xFFFFFFFF
    out = bytearray(size)
    for index in range(size):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        out[index] = (state >> 16) & 0xFF
    return bytes(out)


def test_it_lists_members_without_downloading_the_payloads() -> None:
    # Over 64 KiB so the index search cannot incidentally cover the payloads.
    payloads = {f"IMAGE/images/MANUAL/Manual{index:02d}.jpg": incompressible(index, 20_000) for index in range(20)}
    archive = build_zip(payloads)
    assert len(archive) > 200_000
    calls: list[str] = []

    with RemoteZip(URL, client=serve(archive, calls=calls)) as remote:
        members = remote.members()

    assert len(members) == 20
    assert {member.stem for member in members} == {f"Manual{index:02d}" for index in range(20)}
    # One HEAD for the size, then the tail and the directory. Nothing else.
    assert len(calls) == 3
    # The index is a small fraction of the archive; the payloads were not read.
    assert remote.bytes_fetched < len(archive) // 2


def test_directories_and_apple_double_litter_are_hidden_by_default() -> None:
    """MRAMG's archive carries one __MACOSX entry per real file."""
    archive = build_zip(
        {
            "IMAGE/images/MANUAL/Manual01.jpg": b"a" * 512,
            "__MACOSX/IMAGE/images/MANUAL/._Manual01.jpg": b"resource fork",
            "IMAGE/images/MANUAL/": b"",
        }
    )

    with RemoteZip(URL, client=serve(archive)) as remote:
        assert [member.name for member in remote.members()] == ["IMAGE/images/MANUAL/Manual01.jpg"]
        assert len(remote.members(include_apple_double=True)) == 3


def test_selection_is_by_predicate_so_the_caller_decides_what_it_wants() -> None:
    archive = build_zip(
        {
            "IMAGE/images/MANUAL/Manual01.jpg": b"m" * 256,
            "IMAGE/images/RECIPE/Recipe01.jpg": b"r" * 256,
            "IMAGE/images_info/manual_imgs_collection.json": b"{}",
        }
    )

    with RemoteZip(URL, client=serve(archive)) as remote:
        manual = remote.select(lambda member: member.name.startswith("IMAGE/images/MANUAL/"))

    assert [member.stem for member in manual] == ["Manual01"]


# --------------------------------------------------------------------------- #
# Extracting only what was asked for
# --------------------------------------------------------------------------- #


def test_it_extracts_the_wanted_members_byte_exact() -> None:
    payloads = {
        "IMAGE/images/MANUAL/Manual01.jpg": b"\xff\xd8first image payload" * 100,
        "IMAGE/images/MANUAL/Manual02.jpg": b"\xff\xd8second image payload" * 100,
        "IMAGE/images/RECIPE/Recipe01.jpg": b"not wanted" * 100,
    }
    archive = build_zip(payloads)

    with RemoteZip(URL, client=serve(archive)) as remote:
        wanted = remote.select(lambda member: member.name.startswith("IMAGE/images/MANUAL/"))
        extracted = dict(remote.extract(wanted))

    assert {member.stem: body for member, body in extracted.items()} == {
        "Manual01": payloads["IMAGE/images/MANUAL/Manual01.jpg"],
        "Manual02": payloads["IMAGE/images/MANUAL/Manual02.jpg"],
    }


def test_stored_members_are_handled_as_well_as_deflated() -> None:
    archive = build_zip({"IMAGE/a.bin": b"uncompressed payload" * 50}, compress=False)

    with RemoteZip(URL, client=serve(archive)) as remote:
        members = remote.members()
        extracted = dict(remote.extract(members))

    assert next(iter(extracted.values())) == b"uncompressed payload" * 50


def test_neighbouring_members_are_fetched_as_one_range() -> None:
    """The saving that makes 2,599 images one request instead of 2,599."""
    payloads = {f"IMAGE/images/MANUAL/M{index:03d}.jpg": incompressible(index + 100, 4_000) for index in range(30)}
    archive = build_zip(payloads)
    calls: list[str] = []

    with RemoteZip(URL, client=serve(archive, calls=calls)) as remote:
        wanted = remote.members()
        extracted = dict(remote.extract(wanted))

    assert len(extracted) == 30
    range_calls = [call for call in calls if call.startswith("GET")]
    # Three for the index (HEAD, tail, directory are 1 HEAD + 2 GET), then one
    # coalesced GET for all thirty payloads.
    assert len(range_calls) == 3, calls


def test_a_large_gap_splits_the_fetch_into_separate_ranges() -> None:
    """Coalescing must not pull megabytes of members the caller did not want."""
    payloads = {"IMAGE/images/MANUAL/first.jpg": b"a" * 1024}
    # Incompressible filler so the archive really is large between the two wanted
    # members, rather than deflating to nothing.
    payloads["IMAGE/filler.bin"] = incompressible(7, 5_000_000)
    payloads["IMAGE/images/MANUAL/last.jpg"] = b"z" * 1024
    archive = build_zip(payloads)

    calls: list[str] = []
    with RemoteZip(URL, client=serve(archive, calls=calls), gap_tolerance=1024) as remote:
        wanted = remote.select(lambda member: member.name.startswith("IMAGE/images/MANUAL/"))
        extracted = dict(remote.extract(wanted))

    assert len(extracted) == 2
    payload_calls = [call for call in calls if call.startswith("GET")]
    # Two payload ranges rather than one spanning the filler.
    assert len(payload_calls) == 4, calls
    assert remote.bytes_fetched < len(archive)


def test_plan_bytes_reports_the_cost_before_paying_it() -> None:
    payloads = {f"IMAGE/images/MANUAL/M{index}.jpg": bytes([index]) * 4096 for index in range(10)}
    archive = build_zip(payloads)

    with RemoteZip(URL, client=serve(archive)) as remote:
        wanted = remote.members()
        planned = remote.plan_bytes(wanted)
        before = remote.bytes_fetched
        dict(remote.extract(wanted))
        actually = remote.bytes_fetched - before

    assert planned == actually


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_a_server_that_ignores_range_is_refused_rather_than_indulged() -> None:
    """Otherwise a misconfigured CDN silently pulls the whole 1.53 GiB."""
    archive = build_zip({"IMAGE/a.jpg": b"payload" * 100})

    with pytest.raises(RemoteZipError, match="ignored the Range header"):
        with RemoteZip(URL, client=serve(archive, honour_range=False)) as remote:
            remote.members()


def test_a_corrupted_payload_fails_on_its_crc() -> None:
    """A mis-sliced range must not become a corrupt image reported later.

    Stored rather than deflated, so a flipped byte reaches the CRC check instead of
    failing earlier inside zlib - the point being that the CRC is what catches it.
    """
    name = "IMAGE/a.jpg"
    archive = bytearray(build_zip({name: b"original payload" * 64}, compress=False))
    # Local header is 30 bytes plus the name, so the payload starts after that.
    payload_at = 30 + len(name)
    archive[payload_at + 5] ^= 0xFF

    with RemoteZip(URL, client=serve(bytes(archive))) as remote:
        members = remote.members()
        with pytest.raises(RemoteZipError, match="CRC mismatch"):
            dict(remote.extract(members))


def test_a_deflated_payload_that_is_corrupted_also_fails() -> None:
    """Either zlib rejects it or the CRC does; both are a refusal."""
    name = "IMAGE/a.jpg"
    archive = bytearray(build_zip({name: incompressible(3, 4_000)}))
    archive[30 + len(name) + 40] ^= 0xFF

    with RemoteZip(URL, client=serve(bytes(archive))) as remote:
        members = remote.members()
        with pytest.raises((RemoteZipError, Exception)):
            dict(remote.extract(members))


def test_something_that_is_not_a_zip_is_refused() -> None:
    with pytest.raises(RemoteZipError, match="end-of-central-directory"):
        with RemoteZip(URL, client=serve(b"this is not a zip file" * 100)) as remote:
            remote.members()


def test_members_before_open_is_an_error_rather_than_an_empty_list() -> None:
    remote = RemoteZip(URL, client=serve(build_zip({"a": b"b"})))
    with pytest.raises(RemoteZipError, match="not opened"):
        remote.members()


def test_the_default_gap_tolerance_is_the_measured_one() -> None:
    """1 MiB is what collapses MRAMG's Manual images into a single request."""
    assert DEFAULT_GAP_TOLERANCE == 1_024 * 1_024
