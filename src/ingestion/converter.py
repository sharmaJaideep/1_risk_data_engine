from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import polars as pl

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CsvToParquetConverter:
    """Convert raw CSV files into parquet using Polars lazy execution."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path, compression: str = "zstd") -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.compression = compression

    def _csv_paths(self) -> list[Path]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        return sorted(self.input_dir.rglob("*.csv"))

    def _output_path(self, csv_path: Path) -> Path:
        relative_path = csv_path.relative_to(self.input_dir)
        return self.output_dir / relative_path.with_suffix(".parquet")

    def _convert_file(self, csv_path: Path) -> Path:
        output_path = self._output_path(csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_size = csv_path.stat().st_size
        start_time = perf_counter()

        # Try default encoding first; if a UTF-8-related error occurs, retry with
        # a permissive fallback encoding (latin1) and let convert_all handle
        # skipping on persistent failures.
        try:
            lazy_frame = pl.scan_csv(csv_path)
            lazy_frame.sink_parquet(output_path, compression=self.compression)
        except Exception as exc:
            err_str = str(exc).lower()
            if "utf-8" in err_str or "utf8" in err_str or "invalid utf-8" in err_str:
                logger.warning("Encoding issue detected for %s; retrying with pandas latin1 fallback", csv_path.name)
                # Fallback: read with pandas using latin1 and write with polars
                try:
                    pdf = pd.read_csv(csv_path, encoding="latin1")
                    frame = pl.from_pandas(pdf)
                    frame.write_parquet(output_path, compression=self.compression)
                except Exception:
                    # Let outer handler catch and log
                    raise
            else:
                # Re-raise so convert_all can log and continue
                raise

        elapsed = perf_counter() - start_time
        output_size = output_path.stat().st_size
        reduction = 0.0
        if input_size > 0:
            reduction = 100.0 * (1.0 - (output_size / input_size))

        logger.info(
            "Converted %s -> %s | time=%.2fs | size: %s -> %s | reduction=%.1f%%",
            csv_path.name,
            output_path.name,
            elapsed,
            self._format_bytes(input_size),
            self._format_bytes(output_size),
            reduction,
        )
        return output_path

    def convert_all(self) -> list[Path]:
        csv_paths = self._csv_paths()
        if not csv_paths:
            logger.warning("No CSV files found in %s", self.input_dir)
            return []

        converted_paths: list[Path] = []
        for csv_path in csv_paths:
            try:
                converted_paths.append(self._convert_file(csv_path))
            except Exception as exc:
                # Log the failure and continue with remaining files so a single bad
                # CSV won't stop the whole batch. Keep processing other tables.
                logger.exception("Failed to convert %s; skipping. Error: %s", csv_path, exc)
                continue

        return converted_paths

    @staticmethod
    def _format_bytes(byte_count: int) -> str:
        for unit in ["bytes", "KB", "MB", "GB"]:
            if byte_count < 1024.0 or unit == "GB":
                return f"{byte_count:.1f} {unit}"
            byte_count /= 1024.0
        return f"{byte_count:.1f} bytes"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    logger.info("Starting raw CSV -> parquet conversion")
    converter = CsvToParquetConverter(raw_dir, processed_dir)
    converted = converter.convert_all()
    logger.info("Finished conversion for %d files", len(converted))


if __name__ == "__main__":
    main()
