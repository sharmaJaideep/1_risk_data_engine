import logging

import polars as pl
import pytest

from src.ingestion import converter as conv
from src.ingestion.converter import CsvToParquetConverter


def test_converts_csvs_and_preserves_subdirectories(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    (raw / "nested").mkdir(parents=True)
    (raw / "a.csv").write_text("id,v\n1,x\n2,y\n")
    (raw / "nested" / "b.csv").write_text("id,v\n3,z\n")
    (raw / "ignore.txt").write_text("not a csv")

    paths = CsvToParquetConverter(raw, out).convert_all()

    assert sorted(p.relative_to(out).as_posix() for p in paths) == ["a.parquet", "nested/b.parquet"]
    assert pl.read_parquet(out / "a.parquet").to_dicts() == [{"id": 1, "v": "x"}, {"id": 2, "v": "y"}]


def test_sparse_leading_nulls_are_typed_from_the_whole_file(tmp_path):
    # regression: the default 100-row type sample typed bureau.AMT_ANNUITY as String
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = ["id,amt"] + [f"{i}," for i in range(300)] + ["300,1234.5"]
    (raw / "sparse.csv").write_text("\n".join(rows) + "\n")

    CsvToParquetConverter(raw, tmp_path / "out").convert_all()

    frame = pl.read_parquet(tmp_path / "out" / "sparse.parquet")
    assert frame.schema["amt"] == pl.Float64
    assert frame["amt"].null_count() == 300 and frame["amt"][-1] == 1234.5


def test_non_utf8_file_falls_back_to_latin1(tmp_path, caplog):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "latin.csv").write_bytes(b"id,name\n1,caf\xe9\n")

    with caplog.at_level(logging.WARNING):
        paths = CsvToParquetConverter(raw, tmp_path / "out").convert_all()

    assert [p.name for p in paths] == ["latin.parquet"]
    assert pl.read_parquet(paths[0])["name"][0] == "café"
    assert "Encoding issue" in caplog.text


def test_a_bad_file_is_skipped_and_the_rest_still_convert(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "empty.csv").write_text("")  # no header: cannot be parsed
    (raw / "good.csv").write_text("id\n1\n")

    paths = CsvToParquetConverter(raw, tmp_path / "out").convert_all()

    assert [p.name for p in paths] == ["good.parquet"]


def test_empty_directory_returns_nothing_and_missing_directory_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    assert CsvToParquetConverter(tmp_path / "empty", tmp_path / "out").convert_all() == []
    with pytest.raises(FileNotFoundError):
        CsvToParquetConverter(tmp_path / "missing", tmp_path / "out").convert_all()


@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, "0.0 bytes"), (1023, "1023.0 bytes"), (1024, "1.0 KB"), (5 * 1024**2, "5.0 MB"), (3 * 1024**3, "3.0 GB"), (2048 * 1024**3, "2048.0 GB")],
)
def test_format_bytes(size, expected):
    assert CsvToParquetConverter._format_bytes(size) == expected


def test_module_main_uses_project_raw_and_processed_dirs(monkeypatch):
    calls = {}

    class Fake:
        def __init__(self, raw, processed):
            calls["dirs"] = (raw.parts[-2:], processed.parts[-2:])

        def convert_all(self):
            return []

    monkeypatch.setattr(conv, "CsvToParquetConverter", Fake)
    conv.main()
    assert calls["dirs"] == (("data", "raw"), ("data", "processed"))
