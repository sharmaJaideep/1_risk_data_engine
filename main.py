from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)

STEPS = ["ingest", "drift", "validate", "aggregate", "stability", "vintage", "diagnose"]


def run_step(step: str, data_dir: Path, args: argparse.Namespace) -> None:
    raw_dir, processed_dir = data_dir / "raw", data_dir / "processed"
    logger.info("=== Step: %s ===", step)

    if step == "ingest":
        from src.ingestion.converter import CsvToParquetConverter

        CsvToParquetConverter(raw_dir, processed_dir).convert_all()
    elif step == "drift":
        from src.ingestion.schema_drift import run_drift_report

        report = run_drift_report(processed_dir)
        for row in report.iter_rows(named=True):
            logger.warning("[%s] %s %s.%s: %s -> %s", row["severity"], row["kind"], row["table"], row["column"], row["baseline"], row["observed"])
    elif step == "validate":
        import polars as pl

        from src.ingestion import schema_registry
        from src.ingestion.schema_validator import SchemaValidator

        failed = False
        for table in schema_registry.tables():
            parquet = processed_dir / f"{table}.parquet"
            if not parquet.exists():
                logger.warning("%s: no parquet to validate", table)
                continue
            validator = SchemaValidator.for_table(table)
            errors = validator.validate_frame(pl.scan_parquet(parquet))
            for err in errors:
                logger.error("%s: %s", table, err)
            logger.info("%s (v%s): %s", table, validator.version, "OK" if not errors else f"{len(errors)} violation(s)")
            failed = failed or bool(errors)
        if failed:
            raise SystemExit("Schema validation failed")
    elif step == "aggregate":
        from src.metrics.aggregators import build_master_analytical_record

        build_master_analytical_record(str(processed_dir))
    elif step == "stability":
        from src.metrics.stability import calculate_portfolio_stability

        calculate_portfolio_stability(str(processed_dir / "master_analytical_record.parquet"))
    elif step == "vintage":
        from src.metrics.vintage import build_vintage_reports

        build_vintage_reports(processed_dir)
    elif step == "diagnose":
        from src.agents.diagnostic_agent import DiagnosticAgent

        report = DiagnosticAgent(
            report_path=processed_dir / "portfolio_stability_report.parquet",
            use_llm=not args.no_llm,
        ).run()
        markdown = report.to_markdown()
        out = processed_dir / "diagnostic_brief.md"
        out.write_text(markdown)
        print(markdown)
        logger.info("Diagnostic brief written to %s", out)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Risk Data Engine CLI")
    parser.add_argument("command", choices=[*STEPS, "run-all", "schema-diff"], help="pipeline step to run")
    parser.add_argument("--table", help="schema-diff: table to compare")
    parser.add_argument("--from-version", type=int, help="schema-diff: older version (default: previous)")
    parser.add_argument("--to-version", type=int, help="schema-diff: newer version (default: latest)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM narrative in `diagnose`")
    parser.add_argument("--skip-ingest", action="store_true", help="with run-all, reuse existing parquet files")
    args = parser.parse_args()

    if args.command == "schema-diff":
        from src.ingestion.schema_drift import compare_versions

        if not args.table:
            parser.error("schema-diff requires --table")
        try:
            findings = compare_versions(args.table, args.from_version, args.to_version)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        for f in findings:
            print(f"[{f.severity}] {f.kind} {f.column}: {f.baseline} -> {f.observed}")
        print(f"{len(findings)} difference(s)")
        return

    steps = STEPS if args.command == "run-all" else [args.command]
    if args.skip_ingest:
        steps = [s for s in steps if s != "ingest"]
    for step in steps:
        run_step(step, args.data_dir, args)


if __name__ == "__main__":
    main()
