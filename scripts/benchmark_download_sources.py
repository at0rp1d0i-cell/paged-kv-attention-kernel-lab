#!/usr/bin/env python3
"""Measure package-index latency and capped wheel download throughput."""

from __future__ import annotations

import argparse
import html
import platform
import re
import statistics
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Callable
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

DEFAULT_SAMPLE_MIB = 8
DEFAULT_REPEATS = 2
READ_CHUNK_BYTES = 256 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
USER_AGENT = "paged-kv-source-benchmark/1.0"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    index_url: str
    package: str


@dataclass(frozen=True)
class Sample:
    index_ms: float
    first_byte_ms: float
    download_mib_s: float
    bytes_read: int
    artifact: str


DEFAULT_SOURCES = (
    SourceSpec("tuna-pypi", "https://pypi.tuna.tsinghua.edu.cn/simple", "numpy"),
    SourceSpec("official-pypi", "https://pypi.org/simple", "numpy"),
    SourceSpec("pytorch-cu130", "https://download.pytorch.org/whl/cu130", "torch"),
)


class SimpleIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(html.unescape(href))


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def package_index_url(spec: SourceSpec) -> str:
    package = normalize_package_name(spec.package)
    return f"{spec.index_url.rstrip('/')}/{package}/"


def artifact_name(url: str) -> str:
    return unquote(PurePosixPath(urlsplit(url).path).name)


def _wheel_compatibility_score(filename: str) -> int | None:
    lower = filename.lower()
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cp_tags = re.findall(r"-cp\d{2,3}", lower)

    if f"-{python_tag}" in lower:
        score = 100
    elif "-py3-none-any.whl" in lower:
        score = 80
    elif cp_tags:
        return None
    else:
        score = 20

    system = platform.system().lower()
    if system == "linux":
        if "win_" in lower or "macosx_" in lower:
            return None
        libc_name = platform.libc_ver()[0].lower()
        if libc_name == "glibc" and "musllinux" in lower:
            return None
        if libc_name == "musl" and "manylinux" in lower:
            return None
        if "manylinux" in lower or "linux_" in lower:
            score += 20
    elif system == "darwin":
        if "win_" in lower or "manylinux" in lower or "linux_" in lower:
            return None
        if "macosx_" in lower:
            score += 20
    elif system == "windows":
        if "macosx_" in lower or "manylinux" in lower or "linux_" in lower:
            return None
        if "win_" in lower:
            score += 20

    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        if "aarch64" in lower or "arm64" in lower:
            return None
        if "x86_64" in lower or "amd64" in lower:
            score += 10
    elif machine in {"aarch64", "arm64"}:
        if "x86_64" in lower or "amd64" in lower:
            return None
        if "aarch64" in lower or "arm64" in lower:
            score += 10

    return score


def _wheel_version_key(filename: str) -> tuple[int, int, int, int, int, str]:
    version = filename.split("-", maxsplit=2)[1]
    public_version = version.split("+", maxsplit=1)[0].lower()
    match = re.match(r"(\d+(?:\.\d+)*)(.*)", public_version)
    if match is None:
        return (0, 0, 0, 0, 0, public_version)

    release = [int(part) for part in match.group(1).split(".")]
    padded_release = (release + [0, 0, 0, 0])[:4]
    suffix = match.group(2)
    is_stable = int(not suffix)
    return (*padded_release, is_stable, suffix)


def select_wheel(index_url: str, links: list[str], package: str) -> str:
    normalized_package = normalize_package_name(package)
    candidates: list[tuple[int, tuple[int, int, int, int, int, str], str, str]] = []

    for href in links:
        absolute = urljoin(index_url, href)
        parsed = urlsplit(absolute)
        clean_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
        filename = artifact_name(clean_url)
        if not filename.lower().endswith(".whl"):
            continue
        wheel_package = normalize_package_name(filename.split("-", maxsplit=1)[0])
        if wheel_package != normalized_package:
            continue
        score = _wheel_compatibility_score(filename)
        if score is not None:
            candidates.append((score, _wheel_version_key(filename), filename, clean_url))

    if not candidates:
        raise ValueError(f"no compatible wheel found for {package!r}")

    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _read_with_deadline(response: object, size: int, deadline: float) -> bytes:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("request exceeded the total timeout")

    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        sock.settimeout(remaining)
    return response.read1(size)  # type: ignore[attr-defined, no-any-return]


def fetch_wheel_url(spec: SourceSpec, timeout_s: float) -> tuple[str, float]:
    index_url = package_index_url(spec)
    request = Request(index_url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    started = time.perf_counter()
    deadline = started + timeout_s
    chunks = []
    bytes_read = 0
    with urlopen(request, timeout=timeout_s) as response:
        while bytes_read <= MAX_INDEX_BYTES:
            chunk = _read_with_deadline(response, READ_CHUNK_BYTES, deadline)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    if bytes_read > MAX_INDEX_BYTES:
        raise ValueError(f"package index exceeds {MAX_INDEX_BYTES // (1024 * 1024)} MiB")

    parser = SimpleIndexParser()
    parser.feed(b"".join(chunks).decode("utf-8", errors="replace"))
    return select_wheel(index_url, parser.links, spec.package), elapsed_ms


def download_sample(url: str, sample_bytes: int, timeout_s: float) -> tuple[float, float, int]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
            "Range": f"bytes=0-{sample_bytes - 1}",
        },
    )
    started = time.perf_counter()
    deadline = started + timeout_s
    first_byte_at: float | None = None
    bytes_read = 0

    with urlopen(request, timeout=timeout_s) as response:
        while bytes_read < sample_bytes:
            read_size = 1 if first_byte_at is None else min(
                READ_CHUNK_BYTES, sample_bytes - bytes_read
            )
            chunk = _read_with_deadline(
                response,
                read_size,
                deadline,
            )
            if not chunk:
                break
            if first_byte_at is None:
                first_byte_at = time.perf_counter()
            bytes_read += len(chunk)

    finished = time.perf_counter()
    if first_byte_at is None or bytes_read == 0:
        raise ValueError("artifact response contained no data")

    first_byte_ms = (first_byte_at - started) * 1_000
    stream_seconds = max(finished - first_byte_at, 1e-9)
    download_mib_s = bytes_read / (1024 * 1024) / stream_seconds
    return first_byte_ms, download_mib_s, bytes_read


def measure_source(
    spec: SourceSpec,
    *,
    sample_bytes: int,
    repeats: int,
    timeout_s: float,
    progress: Callable[[str], None] | None = None,
) -> list[Sample]:
    samples = []
    for repeat_idx in range(repeats):
        if progress is not None:
            progress(f"repeat {repeat_idx + 1}/{repeats}: index")
        try:
            wheel_url, index_ms = fetch_wheel_url(spec, timeout_s)
        except Exception as exc:
            raise RuntimeError(f"index request failed: {exc}") from exc
        if progress is not None:
            progress(f"repeat {repeat_idx + 1}/{repeats}: wheel {artifact_name(wheel_url)}")
        try:
            first_byte_ms, download_mib_s, bytes_read = download_sample(
                wheel_url, sample_bytes, timeout_s
            )
        except Exception as exc:
            raise RuntimeError(f"wheel request failed for {artifact_name(wheel_url)}: {exc}") from exc
        samples.append(
            Sample(
                index_ms=index_ms,
                first_byte_ms=first_byte_ms,
                download_mib_s=download_mib_s,
                bytes_read=bytes_read,
                artifact=artifact_name(wheel_url),
            )
        )
    return samples


def parse_source(value: str) -> SourceSpec:
    parts = value.split(",", maxsplit=2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("expected NAME,INDEX_URL,PACKAGE")
    return SourceSpec(parts[0], parts[1], parts[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        help="custom NAME,INDEX_URL,PACKAGE; repeat to compare sources (replaces defaults)",
    )
    parser.add_argument("--sample-mib", type=int, default=DEFAULT_SAMPLE_MIB)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="total timeout for each index or wheel request in seconds",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_mib <= 0 or args.repeats <= 0 or args.timeout <= 0:
        raise SystemExit("--sample-mib, --repeats, and --timeout must be positive")

    sources = tuple(args.source or DEFAULT_SOURCES)
    sample_bytes = args.sample_mib * 1024 * 1024
    rows: list[tuple[SourceSpec, list[Sample] | None, str | None]] = []

    for spec in sources:
        print(f"Measuring {spec.name} ({spec.package})...", flush=True)
        try:
            samples = measure_source(
                spec,
                sample_bytes=sample_bytes,
                repeats=args.repeats,
                timeout_s=args.timeout,
                progress=lambda message: print(f"  {message}", flush=True),
            )
            rows.append((spec, samples, None))
        except Exception as exc:
            rows.append((spec, None, f"{type(exc).__name__}: {exc}"))

    print()
    print(
        f"{'source':<18} {'package':<10} {'index p50':>11} "
        f"{'TTFB p50':>10} {'stream p50':>12}  artifact/status"
    )
    print("-" * 110)
    succeeded = 0
    for spec, samples, error in rows:
        if samples is None:
            print(f"{spec.name:<18} {spec.package:<10} {'FAILED':>35}  {error}")
            continue
        succeeded += 1
        index_ms = statistics.median(sample.index_ms for sample in samples)
        first_byte_ms = statistics.median(sample.first_byte_ms for sample in samples)
        download_mib_s = statistics.median(sample.download_mib_s for sample in samples)
        artifact = samples[-1].artifact
        print(
            f"{spec.name:<18} {spec.package:<10} {index_ms:>8.1f} ms "
            f"{first_byte_ms:>7.1f} ms {download_mib_s:>9.1f} MiB/s  {artifact}"
        )

    print()
    print(
        "Use medians from repeated runs. Compare throughput only for similar artifacts; "
        "resolver and cache costs are outside this probe."
    )
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
