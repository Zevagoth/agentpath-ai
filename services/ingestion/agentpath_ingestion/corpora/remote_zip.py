"""Read selected members out of a zip archive over HTTP, without downloading it.

MRAMG-Bench ships every image for all six of its subsets in one 1.53 GiB
``IMAGE.zip``. Stage 2B1 only needs the Manual subset, and the instruction was to
download the required section rather than the whole thing. A zip keeps its index at
the *end*, so given byte-range support both halves of that are possible:

1. read the last 64 KiB, find the end-of-central-directory record;
2. read the central directory it points at - 3.5 MiB here - and learn every
   member's name, compressed size, CRC and offset;
3. fetch only the byte spans of the members wanted, and inflate them locally.

Measured on this archive: the 2,599 Manual images occupy 132.0 MiB of the 1.53 GiB,
and they sit close enough together that coalescing their ranges with a 1 MiB gap
tolerance turns 2,599 members into **one** request of 133.8 MiB - 1.4% more bytes
than the members themselves. So the saving is 91.8% of the download, and the cost
is one extra request for the index.

Integrity is not taken on trust. Every member carries a CRC-32 in the directory,
so each inflated payload is checked against it. A truncated or mis-sliced range
fails loudly here rather than becoming a corrupt JPEG that some later stage
reports as a broken image.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import httpx
import structlog

logger = structlog.get_logger(__name__)

#: How much of the tail to read when looking for the EOCD record. The record is 22
#: bytes plus a comment of up to 64 KiB, so this is the whole possible range.
EOCD_SEARCH_BYTES = 65_536

#: Local file headers are 30 bytes plus a name and an extra field. Fetching a
#: little past a member's recorded offset means the header can be parsed from the
#: same range as the payload, halving the request count.
LOCAL_HEADER_SLACK = 1_024

#: Two members closer than this are fetched as one range. At 1 MiB the Manual
#: subset collapses to a single request while wasting 1.4% of bytes; at 0 it takes
#: ten requests to save 0.9 MiB, which is the wrong trade on a home connection.
DEFAULT_GAP_TOLERANCE = 1_024 * 1_024

STORED = 0
DEFLATED = 8


class RemoteZipError(RuntimeError):
    """The archive could not be read, or a member did not verify."""


@dataclass(frozen=True, slots=True)
class ZipMember:
    """One entry in the archive's central directory."""

    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int

    @property
    def is_directory(self) -> bool:
        return self.name.endswith("/")

    @property
    def stem(self) -> str:
        """Filename without directories or extension, e.g. ``Manual04_0``."""
        return self.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    @property
    def is_apple_double(self) -> bool:
        """macOS resource-fork litter. This archive carries one per real member."""
        return self.name.startswith("__MACOSX") or self.name.rsplit("/", 1)[-1].startswith("._")


@dataclass(frozen=True, slots=True)
class _Run:
    """A contiguous byte span covering one or more members."""

    start: int
    end: int
    members: tuple[ZipMember, ...]

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class RemoteZip:
    """A zip archive addressed by HTTP range requests.

    The constructor performs no I/O. :meth:`open` reads the index, after which
    :meth:`members` is available and :meth:`extract` can fetch payloads.
    """

    def __init__(
        self,
        url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 300.0,
        gap_tolerance: int = DEFAULT_GAP_TOLERANCE,
    ) -> None:
        self._url = url
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout
        self._gap_tolerance = gap_tolerance
        self._size: int | None = None
        self._members: tuple[ZipMember, ...] = ()
        self._bytes_fetched = 0

    # -- lifecycle ---------------------------------------------------------- #

    def __enter__(self) -> RemoteZip:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    @property
    def http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout, follow_redirects=True)
        return self._client

    @property
    def bytes_fetched(self) -> int:
        """Total payload bytes pulled over the network by this instance."""
        return self._bytes_fetched

    @property
    def archive_size(self) -> int:
        if self._size is None:
            raise RemoteZipError("archive not opened")
        return self._size

    # -- reading the index -------------------------------------------------- #

    def open(self) -> RemoteZip:
        """Read the central directory. Two requests, ~3.5 MiB for this archive."""
        self._size = self._head_size()
        tail_start = max(0, self._size - EOCD_SEARCH_BYTES)
        tail = self._fetch(tail_start, self._size - 1)

        marker = tail.rfind(b"PK\x05\x06")
        if marker < 0:
            raise RemoteZipError("no end-of-central-directory record; not a zip, or truncated")

        entry_count, directory_size, directory_offset = struct.unpack_from("<HII", tail, marker + 10)
        if 0xFFFFFFFF in (directory_size, directory_offset) or entry_count == 0xFFFF:
            # Zip64 puts the real values in a separate record. Unsupported rather
            # than half-supported: guessing here would mis-slice every member.
            raise RemoteZipError("zip64 archive is not supported by this reader")

        directory = self._fetch(directory_offset, directory_offset + directory_size - 1)
        self._members = tuple(_parse_central_directory(directory))
        logger.info(
            "corpus.remote_zip.opened",
            url=self._url,
            archive_bytes=self._size,
            members=len(self._members),
            index_bytes=len(tail) + len(directory),
        )
        return self

    def _head_size(self) -> int:
        """The archive's length in bytes.

        ``x-linked-size`` is preferred because Hugging Face sets it on the redirect
        itself, which saves waiting for the CDN hop. ``content-length`` is the
        fallback and is correct once redirects have been followed.

        No minimum is imposed. An earlier version required the size to exceed the
        EOCD search window, reasoning that a smaller value meant it had read the
        redirect body's length rather than the archive's - but that also rejects
        every archive under 64 KiB, which is a perfectly ordinary zip and is what
        the tests build. The search window is clamped to the archive instead.
        """
        response = self.http.head(self._url)
        response.raise_for_status()
        for header in ("x-linked-size", "content-length"):
            raw = response.headers.get(header)
            if raw and raw.isdigit() and int(raw) > 0:
                return int(raw)
        raise RemoteZipError(f"could not determine the size of {self._url}")

    def members(self, *, include_apple_double: bool = False) -> tuple[ZipMember, ...]:
        if not self._members:
            raise RemoteZipError("archive not opened")
        if include_apple_double:
            return self._members
        return tuple(member for member in self._members if not member.is_directory and not member.is_apple_double)

    def select(self, predicate: Callable[[ZipMember], bool]) -> tuple[ZipMember, ...]:
        return tuple(member for member in self.members() if predicate(member))

    # -- fetching payloads -------------------------------------------------- #

    def extract(self, members: Sequence[ZipMember]) -> Iterator[tuple[ZipMember, bytes]]:
        """Yield ``(member, bytes)`` for each member, fetching coalesced ranges.

        Ordered by archive offset rather than by the caller's order, so the network
        reads forward through the file. Callers that need their own order should
        collect into a mapping.
        """
        wanted = sorted(members, key=lambda member: member.local_offset)
        for run in _coalesce(wanted, self._gap_tolerance):
            blob = self._fetch(run.start, self._clamp(run.end))
            for member in run.members:
                yield member, self._member_from_blob(member, blob, run.start)

    def _clamp(self, end: int) -> int:
        """Keep a run inside the archive.

        Runs are padded by :data:`LOCAL_HEADER_SLACK` so a member's local header
        arrives in the same range as its payload, and for the last member in a small
        archive that padding runs past the end. Clamping here is what makes
        :meth:`plan_bytes` agree with what :meth:`extract` actually fetches.
        """
        return min(end, self.archive_size - 1)

    def _member_from_blob(self, member: ZipMember, blob: bytes, blob_start: int) -> bytes:
        """Slice one member out of a fetched range and verify it."""
        header_at = member.local_offset - blob_start
        if header_at < 0 or header_at + 30 > len(blob):
            raise RemoteZipError(f"range did not cover the local header for {member.name}")
        if blob[header_at : header_at + 4] != b"PK\x03\x04":
            raise RemoteZipError(f"bad local header signature for {member.name}")

        name_length, extra_length = struct.unpack_from("<HH", blob, header_at + 26)
        payload_at = header_at + 30 + name_length + extra_length
        payload = blob[payload_at : payload_at + member.compressed_size]
        if len(payload) != member.compressed_size:
            raise RemoteZipError(f"range covered {len(payload)} of {member.compressed_size} bytes for {member.name}")

        if member.method == DEFLATED:
            data = zlib.decompress(payload, -15)
        elif member.method == STORED:
            data = payload
        else:
            raise RemoteZipError(f"unsupported compression method {member.method} for {member.name}")

        # The directory's CRC is the archive's own claim about the bytes. Checking
        # it turns a mis-sliced range into an error here instead of a corrupt file
        # that some later stage reports as a broken image.
        if zlib.crc32(data) & 0xFFFFFFFF != member.crc32:
            raise RemoteZipError(f"CRC mismatch for {member.name}; the fetched range is wrong")
        if member.uncompressed_size and len(data) != member.uncompressed_size:
            raise RemoteZipError(f"{member.name} inflated to {len(data)} bytes, expected {member.uncompressed_size}")
        return data

    def _fetch(self, start: int, end: int) -> bytes:
        response = self.http.get(self._url, headers={"Range": f"bytes={start}-{end}"})
        if response.status_code not in (200, 206):
            raise RemoteZipError(f"range request returned HTTP {response.status_code}")
        if response.status_code == 200:
            # A server that ignored Range sent the whole archive; refuse rather
            # than silently pulling 1.5 GiB.
            raise RemoteZipError("server ignored the Range header; refusing a full download")
        body = response.content
        self._bytes_fetched += len(body)
        return body

    def plan_bytes(self, members: Sequence[ZipMember]) -> int:
        """Bytes that :meth:`extract` would fetch for *members*, before fetching.

        Exact, not an estimate: the same coalescing and the same clamping, so a
        caller can report the cost of a download before committing to it.
        """
        wanted = sorted(members, key=lambda member: member.local_offset)
        return sum(self._clamp(run.end) - run.start + 1 for run in _coalesce(wanted, self._gap_tolerance))


def _parse_central_directory(directory: bytes) -> Iterable[ZipMember]:
    cursor = 0
    limit = len(directory)
    while cursor + 46 <= limit:
        if directory[cursor : cursor + 4] != b"PK\x01\x02":
            break
        (
            method,
            _time,
            _date,
            crc32,
            compressed,
            uncompressed,
            name_length,
            extra_length,
            comment_length,
        ) = struct.unpack_from("<10xHHHIIIHHH", directory, cursor)
        local_offset = struct.unpack_from("<I", directory, cursor + 42)[0]
        name = directory[cursor + 46 : cursor + 46 + name_length].decode("utf-8", "replace")
        yield ZipMember(
            name=name,
            method=method,
            crc32=crc32,
            compressed_size=compressed,
            uncompressed_size=uncompressed,
            local_offset=local_offset,
        )
        cursor += 46 + name_length + extra_length + comment_length


def _coalesce(members: Sequence[ZipMember], gap_tolerance: int) -> list[_Run]:
    """Group offset-sorted members into as few byte ranges as the gaps allow."""
    if not members:
        return []

    runs: list[_Run] = []
    batch: list[ZipMember] = [members[0]]
    start = members[0].local_offset
    end = _member_end(members[0])

    for member in members[1:]:
        if member.local_offset - end > gap_tolerance:
            runs.append(_Run(start=start, end=end, members=tuple(batch)))
            batch = [member]
            start = member.local_offset
            end = _member_end(member)
            continue
        batch.append(member)
        end = max(end, _member_end(member))

    runs.append(_Run(start=start, end=end, members=tuple(batch)))
    return runs


def _member_end(member: ZipMember) -> int:
    """Last byte that could belong to *member*, allowing for its local header."""
    return member.local_offset + LOCAL_HEADER_SLACK + member.compressed_size


__all__ = [
    "DEFAULT_GAP_TOLERANCE",
    "RemoteZip",
    "RemoteZipError",
    "ZipMember",
]
