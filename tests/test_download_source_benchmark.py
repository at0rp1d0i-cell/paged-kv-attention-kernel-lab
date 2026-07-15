import argparse
import importlib.util
import platform
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_download_sources.py"
SPEC = importlib.util.spec_from_file_location("benchmark_download_sources", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SimpleIndexParser = MODULE.SimpleIndexParser
SourceSpec = MODULE.SourceSpec
package_index_url = MODULE.package_index_url
parse_source = MODULE.parse_source
select_wheel = MODULE.select_wheel


def test_simple_index_parser_collects_and_unescapes_links() -> None:
    parser = SimpleIndexParser()
    parser.feed('<a href="files/numpy.whl?x=1&amp;y=2#sha256=abc">numpy</a>')

    assert parser.links == ["files/numpy.whl?x=1&y=2#sha256=abc"]


def test_select_wheel_prefers_current_python_and_architecture() -> None:
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        platform_tag = "manylinux_2_28_aarch64" if machine in {"aarch64", "arm64"} else "manylinux_2_28_x86_64"
        wrong_platform = "win_amd64"
    elif system == "darwin":
        platform_tag = "macosx_14_0_arm64" if machine in {"aarch64", "arm64"} else "macosx_14_0_x86_64"
        wrong_platform = "manylinux_2_28_x86_64"
    else:
        platform_tag = "win_amd64"
        wrong_platform = "manylinux_2_28_x86_64"

    compatible = f"numpy-2.13.0-{python_tag}-{python_tag}-{platform_tag}.whl"
    links = [
        f"numpy-2.9.0-{python_tag}-{python_tag}-{platform_tag}.whl",
        f"numpy-2.14.0-{python_tag}-{python_tag}-{wrong_platform}.whl",
        f"numpy-2.15.0-cp311-cp311-{platform_tag}.whl",
        compatible + "#sha256=abc",
    ]

    selected = select_wheel("https://example.com/simple/numpy/", links, "numpy")

    assert selected == f"https://example.com/simple/numpy/{compatible}"


def test_source_parser_and_index_url() -> None:
    source = parse_source("mirror,https://example.com/simple/,Num_Py")

    assert source == SourceSpec("mirror", "https://example.com/simple/", "Num_Py")
    assert package_index_url(source) == "https://example.com/simple/num-py/"


def test_source_parser_rejects_incomplete_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="NAME,INDEX_URL,PACKAGE"):
        parse_source("mirror,https://example.com/simple")
