from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import polars as pl
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"

# Roll-rate thresholds: monthly share of accounts that move into / further into delinquency.
# Built-in fallbacks; config/thresholds.yaml overrides these, per source.
DEFAULT_ROLL_THRESHOLDS = {
    "roll_in_max": 0.05,  # CURRENT -> any DPD bucket
    "forward_max": 0.15,  # DPD_1_30 -> DPD_31_60 or worse
}
# Vintage: compare the newest cohort to the trailing cohorts at a fixed months-on-book.
VINTAGE_MOB = 12
VINTAGE_MIN_ACCOUNTS = 1000
VINTAGE_TRAILING = 4
VINTAGE_WORSENING_RATIO = 1.25
VINTAGE_MIN_BAD = 30  # below this many bad accounts in the latest cohort, a ratio is just noise

_SEVERITY = {"ok": 0, "warning": 1, "critical": 2}
_DPD_BUCKETS = ("DPD_1_30", "DPD_31_60", "DPD_61_90", "DPD_91_120", "DPD_120_PLUS")


@dataclass
class DiagnosticReport:
    """Structure for autonomous diagnostics."""

    status: str
    findings: list[str]
    recommendations: list[str]
    narrative: str | None = None
    breaches: list[dict[str, Any]] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# Portfolio Stability Brief", "", f"**Status:** {self.status.upper()}", ""]
        if self.narrative:
            lines += ["## Summary", "", self.narrative, ""]
        lines += ["## Findings", ""] + [f"- {f}" for f in self.findings] + [""]
        lines += ["## Recommendations", ""] + [f"- {r}" for r in self.recommendations] + [""]
        return "\n".join(lines)


class DiagnosticAgent:
    """Inspect the PSI stability report, flag breaches and draft a root-cause brief.

    Breach detection is deterministic. If an LLM is available (``use_llm`` and an
    ``ANTHROPIC_API_KEY``), it additionally writes a narrative root-cause summary;
    otherwise the rule-based findings stand on their own.
    """

    def __init__(
        self,
        context: dict[str, Any] | None = None,
        report_path: str | Path = "data/processed/portfolio_stability_report.parquet",
        schema_drift_path: str | Path | None = None,
        thresholds_path: str | Path = "config/thresholds.yaml",
        roll_rate_path: str | Path | None = None,
        vintage_path: str | Path | None = None,
        use_llm: bool = True,
        model: str = DEFAULT_MODEL,
        top_n: int = 10,
    ) -> None:
        self.context = context or {}
        self.report_path = Path(report_path)
        base = self.report_path.parent
        self.schema_drift_path = Path(schema_drift_path) if schema_drift_path else base / "schema_drift_report.parquet"
        self.roll_rate_path = Path(roll_rate_path) if roll_rate_path else base / "roll_rate_matrix.parquet"
        self.vintage_path = Path(vintage_path) if vintage_path else base / "vintage_curves.parquet"
        self.use_llm = use_llm
        self.model = model
        self.top_n = top_n
        self.thresholds_path = Path(thresholds_path)
        self._roll_config = self._load_roll_config()

    def run(self) -> DiagnosticReport:
        if not self.report_path.exists():
            return DiagnosticReport(
                status="error",
                findings=[f"Stability report not found at {self.report_path}"],
                recommendations=["Run `python main.py run-all` (or the `stability` step) first"],
            )

        report = (
            pl.read_parquet(self.report_path).drop_nulls("psi_score")
            .filter(~pl.col("feature_name").str.starts_with("SK_ID"))  # identifiers, not drift features
            .sort("psi_score", descending=True)
        )
        significant = report.filter(pl.col("psi_score") >= PSI_SIGNIFICANT)
        moderate = report.filter((pl.col("psi_score") >= PSI_MODERATE) & (pl.col("psi_score") < PSI_SIGNIFICANT))

        status = "critical" if significant.height else "warning" if moderate.height else "ok"
        findings = [
            f"{report.height} features scored: {significant.height} significant drift "
            f"(PSI >= {PSI_SIGNIFICANT}), {moderate.height} moderate shift, "
            f"{report.height - significant.height - moderate.height} stable"
        ]
        breaches = report.filter(pl.col("psi_score") >= PSI_MODERATE).head(self.top_n).to_dicts()
        findings += [f"{b['feature_name']}: PSI={b['psi_score']:.4f} ({b['drift_status']})" for b in breaches]

        recommendations = self._recommendations(status, breaches)

        for check in (self._check_schema_drift, self._check_roll_rates, self._check_vintage):
            sev, extra_findings, extra_recs = check()
            findings += extra_findings
            recommendations += [r for r in extra_recs if r not in recommendations]
            if _SEVERITY[sev] > _SEVERITY[status]:
                status = sev
        if status != "ok":
            recommendations = [r for r in recommendations if not r.startswith("No action required")]

        narrative = self._narrative(findings) if self.use_llm and status != "ok" else None
        return DiagnosticReport(status, findings, recommendations, narrative, breaches)

    def _check_schema_drift(self) -> tuple[str, list[str], list[str]]:
        if not self.schema_drift_path.exists():
            return "ok", ["Schema drift report not found; skipped (run the `drift` step)"], []
        drift = pl.read_parquet(self.schema_drift_path)
        breaking = drift.filter(pl.col("severity") == "breaking")
        warnings = drift.filter(pl.col("severity") == "warning")
        findings = [
            f"Schema drift: {breaking.height} breaking, {warnings.height} warning, "
            f"{drift.height - breaking.height - warnings.height} informational"
        ]
        for row in pl.concat([breaking, warnings]).head(self.top_n).iter_rows(named=True):
            column = f".{row['column']}" if row["column"] else ""
            findings.append(
                f"[{row['severity']}] {row['kind']} {row['table']}{column}: {row['baseline']} -> {row['observed']}"
            )
        if breaking.height:
            return (
                "critical",
                findings,
                ["Resolve breaking schema drift before running metrics: fix the upstream feed or version the schema deliberately"],
            )
        if warnings.height:
            return "warning", findings, ["Review schema warnings and update config/schemas/ if the change is intended"]
        return "ok", findings, []

    def _load_roll_config(self) -> dict[str, Any]:
        if not self.thresholds_path.exists():
            logger.info("%s not found; using built-in roll-rate thresholds", self.thresholds_path)
            return {}
        return (yaml.safe_load(self.thresholds_path.read_text()) or {}).get("roll_rates", {})

    def roll_thresholds(self, source: str) -> dict[str, float]:
        """Thresholds for a source: built-ins, then the file's default, then the source's own overrides."""
        return {
            **DEFAULT_ROLL_THRESHOLDS,
            **self._roll_config.get("default", {}),
            **self._roll_config.get("sources", {}).get(source, {}),
        }

    @staticmethod
    def _per_source(
        path: Path, missing_msg: str, check: Callable[[pl.DataFrame, str], tuple[str, list[str], list[str]]]
    ) -> tuple[str, list[str], list[str]]:
        """Run `check` on each source's rows (files without a `source` column are treated as bureau)."""
        if not path.exists():
            return "ok", [missing_msg], []
        frame = pl.read_parquet(path)
        if "source" not in frame.columns:
            frame = frame.with_columns(pl.lit("bureau").alias("source"))
        severity, findings, recs = "ok", [], []
        for source in frame["source"].unique(maintain_order=True):
            sev, f, r = check(frame.filter(pl.col("source") == source), source)
            findings += [f"[{source}] {line}" for line in f]
            recs += [x for x in r if x not in recs]
            if _SEVERITY[sev] > _SEVERITY[severity]:
                severity = sev
        return severity, findings, recs

    def _check_roll_rates(self) -> tuple[str, list[str], list[str]]:
        return self._per_source(
            self.roll_rate_path, "Roll-rate matrix not found; skipped (run the `vintage` step)", self._roll_rates_for
        )

    def _check_vintage(self) -> tuple[str, list[str], list[str]]:
        return self._per_source(
            self.vintage_path, "Vintage curves not found; skipped (run the `vintage` step)", self._vintage_for
        )

    def _roll_rates_for(self, rr: pl.DataFrame, source: str) -> tuple[str, list[str], list[str]]:

        def share(from_bucket: str, to_buckets: tuple[str, ...]) -> float:
            sel = rr.filter(pl.col("from_bucket") == from_bucket)
            return float(sel.filter(pl.col("to_bucket").is_in(to_buckets))["roll_rate"].sum())

        limits = self.roll_thresholds(source)
        roll_in_max, forward_max = limits["roll_in_max"], limits["forward_max"]
        roll_in = share("CURRENT", _DPD_BUCKETS)
        forward = share("DPD_1_30", _DPD_BUCKETS[1:])
        cure = share("DPD_1_30", ("CURRENT",))
        findings = [
            f"Roll rates: CURRENT->delinquent {roll_in:.1%}, 1-30 DPD->31+ DPD {forward:.1%}, 1-30 DPD cures to CURRENT {cure:.1%}"
        ]
        recs: list[str] = []
        severity = "ok"
        if roll_in > roll_in_max:
            severity = "warning"
            findings.append(f"CURRENT->delinquent roll-in {roll_in:.1%} exceeds {roll_in_max:.1%} threshold")
            recs.append("Review origination and early-account management: too many current accounts are slipping into delinquency")
        if forward > forward_max:
            severity = "warning"
            findings.append(f"1-30 DPD forward roll {forward:.1%} exceeds {forward_max:.1%} threshold")
            recs.append("Strengthen early-stage collections: early delinquents are not curing")
        return severity, findings, recs

    @staticmethod
    def _vintage_for(curves: pl.DataFrame, source: str) -> tuple[str, list[str], list[str]]:
        # The oldest cohort absorbs accounts truncated by the 96-month history window; exclude it.
        oldest = curves["cohort_start_month"].min()
        v = curves.filter(
            (pl.col("mob") == VINTAGE_MOB)
            & (pl.col("accounts_observed") >= VINTAGE_MIN_ACCOUNTS)
            & (pl.col("cohort_start_month") > oldest)
        ).sort("cohort_start_month")
        if v.height <= VINTAGE_TRAILING:
            return "ok", [f"Too few cohorts with MOB {VINTAGE_MOB} data for a vintage trend check"], []

        latest = v.tail(1).row(0, named=True)
        trailing = v.slice(v.height - 1 - VINTAGE_TRAILING, VINTAGE_TRAILING)["cumulative_bad_rate"].mean()
        rate = latest["cumulative_bad_rate"]
        ratio = rate / trailing if trailing else float("inf") if rate else 1.0
        finding = (
            f"Vintage: cohort starting month {latest['cohort_start_month']} has a {rate:.2%} cumulative bad rate "
            f"at MOB {VINTAGE_MOB} vs {trailing:.2%} for the prior {VINTAGE_TRAILING} cohorts ({ratio:.2f}x)"
        )
        if ratio > VINTAGE_WORSENING_RATIO and latest["accounts_bad"] < VINTAGE_MIN_BAD:
            return "ok", [finding + f" - not flagged: only {latest['accounts_bad']} bad accounts"], []
        if ratio > VINTAGE_WORSENING_RATIO:
            return (
                "warning",
                [finding + " - worsening"],
                ["Audit recent originations (policy, channel, population mix): newer vintages are defaulting faster"],
            )
        return "ok", [finding], []

    @staticmethod
    def _recommendations(status: str, breaches: list[dict[str, Any]]) -> list[str]:
        if status == "ok":
            return ["No action required; continue routine monitoring"]
        recs = ["Investigate upstream changes to the drifted features' source tables before retraining"]
        if any(b["psi_score"] >= PSI_SIGNIFICANT for b in breaches):
            recs.append("Review the affected features' use in scorecards/models; consider recalibration or retraining")
        recs.append("Re-run stability on additional cohorts to confirm the shift is persistent")
        return recs

    def _narrative(self, findings: list[str]) -> str | None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.info("ANTHROPIC_API_KEY not set; skipping LLM narrative")
            return None
        try:
            from pydantic_ai import Agent

            agent = Agent(
                self.model,
                system_prompt=(
                    "You are a credit-risk model monitoring analyst. Given PSI stability findings "
                    "(baseline = older 70% of the portfolio, target = recent 30%), write a concise "
                    "root-cause brief: likely drivers of drift, business impact, and next steps. "
                    "Do not invent numbers beyond those provided."
                ),
            )
            return agent.run_sync("\n".join(findings)).output
        except Exception as exc:  # LLM is optional; never fail the diagnostic over it
            logger.warning("LLM narrative failed: %s", exc)
            return None
