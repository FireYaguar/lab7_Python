import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex import __version__
from findex.cli import app

runner = CliRunner()


def test_help_and_version() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0 and "search" in res.output and "index" in res.output
    assert __version__ in runner.invoke(app, ["--version"]).output


def test_index_then_search(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ix.bin"
    res = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out), "--positions"])
    assert res.exit_code == 0, res.output
    assert out.is_file() and "Index(terms=" in res.stdout
    res = runner.invoke(app, ["search", str(out), '"event loop"', "-k", "3"])
    assert res.exit_code == 0 and "1 documents match" in res.stdout


def test_search_json_is_pipeable(index_file: Path) -> None:
    res = runner.invoke(app, ["search", str(index_file), "python", "--json", "--scorer", "tfidf"])
    assert res.exit_code == 0
    rows = [json.loads(line) for line in res.stdout.splitlines()]
    assert {r["path"] for r in rows} == {"a.txt", "b.txt", "d.txt"}
    assert rows == sorted(rows, key=lambda r: r["score"], reverse=True)


def test_stats(corpus_dir: Path) -> None:
    res = runner.invoke(app, ["stats", str(corpus_dir), "--top", "3"])
    assert res.exit_code == 0 and "vocabulary" in res.stdout and "filler" in res.stdout


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (["search", "nope.bin", "x"], 1, "index file not found"),
        (["index", "no/such/dir", "--out", "x.bin"], 1, "corpus not found"),
        (["stats", "no/such/dir"], 1, "corpus not found"),
    ],
)
def test_expected_errors_are_one_line(args: list[str], code: int, message: str) -> None:
    res = runner.invoke(app, args)
    assert res.exit_code == code
    assert message in res.stderr and "Traceback" not in res.output + res.stderr


def test_bad_query_exit_code(index_file: Path) -> None:
    res = runner.invoke(app, ["search", str(index_file), "(python"])
    assert res.exit_code == 2 and "bad query" in res.stderr


def test_verbose_logs_go_to_stderr(index_file: Path) -> None:
    res = runner.invoke(app, ["-vv", "search", str(index_file), "python", "--json"])
    assert "search took" in res.stderr and "search took" not in res.stdout


@pytest.mark.slow
def test_full_corpus_end_to_end(tmp_path: Path) -> None:
    data = Path("data")
    if not data.is_dir():
        pytest.skip("corpus not downloaded (scripts/get_corpus.sh)")
    out = tmp_path / "ix.bin"
    assert runner.invoke(app, ["index", str(data), "--out", str(out), "--positions"]).exit_code == 0
    res = runner.invoke(app, ["search", str(out), '"event loop"', "--json"])
    assert "asyncio" in res.stdout


def test_serve_without_index_is_one_line_error(tmp_path: Path) -> None:
    res = runner.invoke(app, ["serve", "--index", str(tmp_path / "none.bin")])
    assert res.exit_code == 1 and "index file not found" in res.stderr
