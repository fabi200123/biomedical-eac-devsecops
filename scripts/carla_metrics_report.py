#!/usr/bin/env python3
"""
Generate CARLA operational metrics for publications.

This tool can pull Prometheus time series, summarize local CSV logs, and emit
both ready-to-plot artifacts (CSVs, PNGs) and a LaTeX table snippet for papers.

Examples:
    python scripts/carla_metrics_report.py \
        --prom-url http://prometheus.monitoring:9090 \
        --start 2024-05-01T00:00:00Z \
        --end 2024-05-07T00:00:00Z \
        --step 5m \
        --rollouts-csv data/carla_runs/rollouts.csv \
        --decisions-csv data/carla_runs/decisions.csv \
        --outdir reports/carla_may
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile CARLA metrics for analysis/papers.")
    parser.add_argument("--prom-url", help="Base URL for the Prometheus server (e.g. http://prom:9090).")
    parser.add_argument("--start", help="Query start time (RFC3339, e.g. 2024-05-01T00:00:00Z).")
    parser.add_argument("--end", help="Query end time (RFC3339).")
    parser.add_argument("--step", default="5m", help="Query resolution/step (default: 5m).")
    parser.add_argument("--outdir", default="reports/carla_metrics", help="Output directory.")
    parser.add_argument("--rollouts-csv", help="Local rollouts CSV (optional).")
    parser.add_argument("--decisions-csv", help="Local decisions CSV (optional).")
    parser.add_argument("--timezone", default="UTC", help="Timezone label for plots (default: UTC).")
    parser.add_argument("--dry-run", action="store_true", help="Parse arguments and exit.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prom_query_range(
    base_url: str,
    query: str,
    start: str,
    end: str,
    step: str,
) -> pd.DataFrame:
    """Query Prometheus HTTP API for a range vector."""
    api = base_url.rstrip("/") + "/api/v1/query_range"
    resp = requests.get(
        api,
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")

    series = payload["data"]["result"]
    if not series:
        return pd.DataFrame(columns=["ts", "value"])

    frames: List[pd.DataFrame] = []
    for entry in series:
        metric_labels = entry.get("metric", {})
        df = pd.DataFrame(entry["values"], columns=["ts", "value"])
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df["value"] = df["value"].astype(float)
        for key, val in metric_labels.items():
            df[f"label_{key}"] = val
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def export_prometheus_metrics(
    prom_url: str,
    start: str,
    end: str,
    step: str,
    outdir: Path,
) -> Dict[str, pd.DataFrame]:
    """Collect required Prometheus metrics and write raw CSV files."""
    results: Dict[str, pd.DataFrame] = {}

    queries = {
        "headroom": "carla_headroom_ratio",
        "decisions_rate": "sum(rate(carla_decisions_total[5m])) by (status)",
        "rollout_mean": (
            "sum(rate(carla_rollout_duration_seconds_sum[5m])) "
            "/ sum(rate(carla_rollout_duration_seconds_count[5m]))"
        ),
    }

    for name, query in queries.items():
        df = prom_query_range(prom_url, query, start, end, step)
        results[name] = df
        df.to_csv(outdir / f"raw_prom_{name}.csv", index=False)

    # Quantiles
    quantiles = {
        "p50": 0.5,
        "p90": 0.9,
        "p95": 0.95,
    }
    quantile_frames: List[pd.DataFrame] = []
    for label, quantile in quantiles.items():
        q_expr = (
            f"histogram_quantile({quantile}, "
            "sum(rate(carla_rollout_duration_seconds_bucket[5m])) by (le))"
        )
        df = prom_query_range(prom_url, q_expr, start, end, step)
        if df.empty:
            continue
        df = df[["ts", "value"]].rename(columns={"value": label})
        quantile_frames.append(df)

    quantiles_path = outdir / "summary_rollout_quantiles.csv"
    if quantile_frames:
        merged = quantile_frames[0]
        for frame in quantile_frames[1:]:
            merged = merged.merge(frame, on="ts", how="outer")
        merged = merged.sort_values("ts")
        results["rollout_quantiles"] = merged
        merged.to_csv(quantiles_path, index=False)
    else:
        results["rollout_quantiles"] = pd.DataFrame(columns=["ts", "p50", "p90", "p95"])
        quantiles_path.write_text("ts,p50,p90,p95\n", encoding="utf-8")

    return results


def summarize_headroom(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if df.empty:
        summary = pd.DataFrame(columns=["metric", "value"])
        summary.to_csv(out_path, index=False)
        return summary
    values = df["value"].astype(float)
    stats = pd.DataFrame(
        {
            "metric": ["mean", "median", "min", "max", "std"],
            "value": [
                values.mean(),
                values.median(),
                values.min(),
                values.max(),
                values.std(ddof=1) if len(values) > 1 else 0.0,
            ],
        }
    )
    stats.to_csv(out_path, index=False)
    return stats


def summarize_decision_rates(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if df.empty:
        summary = pd.DataFrame(columns=["status", "mean_rate"])
        summary.to_csv(out_path, index=False)
        return summary
    if "label_status" not in df.columns:
        df["label_status"] = "unknown"
    grouped = df.groupby("label_status")["value"].mean().reset_index()
    grouped = grouped.rename(columns={"label_status": "status", "value": "mean_rate"})
    grouped.to_csv(out_path, index=False)
    return grouped


def plot_headroom_timeseries(df: pd.DataFrame, out_path: Path, timezone: str) -> None:
    plt.figure(figsize=(10, 4))
    if df.empty:
        plt.text(0.5, 0.5, "No Prometheus data available", ha="center", va="center")
        plt.axis("off")
    else:
        data = df.copy()
        data["ts_local"] = data["ts"].dt.tz_convert(timezone)
        plt.plot(data["ts_local"], data["value"], linewidth=1.5)
        plt.title("CARLA Headroom Ratio")
        plt.xlabel(f"Time ({timezone})")
        plt.ylabel("Headroom")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_decisions_area(df: pd.DataFrame, out_path: Path, timezone: str) -> None:
    plt.figure(figsize=(10, 4))
    if df.empty:
        plt.text(0.5, 0.5, "No Prometheus data available", ha="center", va="center")
        plt.axis("off")
    else:
        data = df.copy()
        data["ts_local"] = data["ts"].dt.tz_convert(timezone)
        pivot = (
            data.pivot_table(
                index="ts_local",
                columns="label_status",
                values="value",
                aggfunc="mean",
            )
            .fillna(0.0)
            .sort_index()
        )
        if pivot.empty:
            plt.text(0.5, 0.5, "No Prometheus data available", ha="center", va="center")
            plt.axis("off")
        else:
            plt.stackplot(pivot.index, pivot.T.values, labels=pivot.columns)
            plt.title("CARLA Decisions Rate by Status")
            plt.xlabel(f"Time ({timezone})")
            plt.ylabel("Decisions per second")
            plt.legend(loc="upper left")
            plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rollout_histograms(rollouts_df: pd.DataFrame, outdir: Path) -> Dict[str, float]:
    if rollouts_df.empty:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No rollout data available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(outdir / "fig_rollout_hist.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No rollout data available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(outdir / "fig_rollout_ecdf.png")
        plt.close()
        return {}
    values = rollouts_df["rollout_seconds"].astype(float)
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20)
    plt.title("Rollout Durations")
    plt.xlabel("Seconds")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "fig_rollout_hist.png")
    plt.close()

    xs = sorted(values)
    if not xs:
        return {}
    ys = [i / len(xs) for i in range(1, len(xs) + 1)]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.title("Rollout Duration ECDF")
    plt.xlabel("Seconds")
    plt.ylabel("ECDF")
    plt.tight_layout()
    plt.savefig(outdir / "fig_rollout_ecdf.png")
    plt.close()

    stats = {
        "n": len(xs),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
        "p95": float(values.quantile(0.95)),
    }
    return stats


def load_rollouts_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "rollout_seconds" in df.columns:
        return df
    for col in df.columns:
        if "rollout" in col.lower():
            df = df.rename(columns={col: "rollout_seconds"})
            return df
    raise ValueError("Could not infer rollout column in CSV.")


def load_decisions_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def latex_table(
    headroom_summary: pd.DataFrame,
    rollout_stats: Dict[str, float],
    decisions_summary: pd.DataFrame,
) -> str:
    if not headroom_summary.empty and {"metric", "value"}.issubset(headroom_summary.columns):
        mean_series = headroom_summary.loc[headroom_summary["metric"] == "mean", "value"]
        std_series = headroom_summary.loc[headroom_summary["metric"] == "std", "value"]
        headroom_mean = float(mean_series.iloc[0]) if not mean_series.empty else math.nan
        headroom_std = float(std_series.iloc[0]) if not std_series.empty else math.nan
    else:
        headroom_mean = math.nan
        headroom_std = math.nan

    p90 = rollout_stats.get("p90", math.nan)
    p95 = rollout_stats.get("p95", math.nan)
    n_rollouts = rollout_stats.get("n", 0)

    decisions_rows = []
    if not decisions_summary.empty and {"status", "mean_rate"}.issubset(decisions_summary.columns):
        for _, row in decisions_summary.iterrows():
            decisions_rows.append(f"{row['status']} & {row['mean_rate']:.3f} \\\\")
    decisions_block = "\n".join(decisions_rows) if decisions_rows else "N/A & -- \\\\"

    latex = (
        "\\begin{table}[!t]\n"
        "  \\centering\n"
        "  \\caption{CARLA runtime metrics summary.}\n"
        "  \\label{tab:carla-metrics}\n"
        "  \\begin{tabular}{ll}\n"
        "    \\toprule\n"
        "    Metric & Value \\\\\n"
        "    \\midrule\n"
        f"    Headroom mean $\\pm$ std & {headroom_mean:.2f} $\\pm$ {headroom_std:.2f} \\\\\n"
        f"    Rollout p90 / p95 (n={int(n_rollouts)}) & {p90:.2f} / {p95:.2f} s \\\\\n"
        "    Decisions rate (status) & \\\\\n"
        f"    \\multicolumn{{2}}{{l}}{{\\begin{{tabular}}{{ll}}\n{decisions_block}\n    \\end{{tabular}}}} \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    return latex


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    prom_results: Dict[str, pd.DataFrame] = {}
    headroom_summary = pd.DataFrame(columns=["metric", "value"])
    decisions_summary = pd.DataFrame(columns=["status", "mean_rate"])
    rollout_quantiles = pd.DataFrame(columns=["ts", "p50", "p90", "p95"])

    if args.prom_url:
        if not (args.start and args.end):
            raise SystemExit("--start and --end are required when --prom-url is provided.")
        prom_results = export_prometheus_metrics(
            args.prom_url, args.start, args.end, args.step, outdir
        )
        headroom_summary = summarize_headroom(
            prom_results.get("headroom", pd.DataFrame()),
            outdir / "summary_headroom.csv",
        )
        decisions_summary = summarize_decision_rates(
            prom_results.get("decisions_rate", pd.DataFrame()),
            outdir / "summary_decisions_rate.csv",
        )
        rollout_quantiles = prom_results.get("rollout_quantiles", pd.DataFrame())

    else:
        # Create empty placeholder CSVs for expected outputs
        (outdir / "summary_headroom.csv").write_text("metric,value\n", encoding="utf-8")
        (outdir / "summary_decisions_rate.csv").write_text("status,mean_rate\n", encoding="utf-8")
        (outdir / "summary_rollout_quantiles.csv").write_text("ts,p50,p90,p95\n", encoding="utf-8")

    plot_headroom_timeseries(
        prom_results.get("headroom", pd.DataFrame(columns=["ts", "value"])),
        outdir / "fig_headroom_ts.png",
        args.timezone,
    )
    plot_decisions_area(
        prom_results.get("decisions_rate", pd.DataFrame(columns=["ts", "value", "label_status"])),
        outdir / "fig_decisions_area.png",
        args.timezone,
    )

    rollout_stats: Dict[str, float] = {}
    if args.rollouts_csv:
        rollouts_df = load_rollouts_csv(Path(args.rollouts_csv))
        rollout_stats = plot_rollout_histograms(rollouts_df, outdir)
        # Export enriched rollouts summary for reproducibility
        rollouts_df.to_csv(outdir / "rollouts_local_copy.csv", index=False)
        if rollout_stats:
            n = max(rollout_stats["n"], 1)
            ci95 = 1.96 * (rollout_stats["std"] / math.sqrt(n)) if n > 1 else 0.0
            summary_row = {
                "n": rollout_stats["n"],
                "mean_rollout_s": rollout_stats["mean"],
                "std_s": rollout_stats["std"],
                "ci95_lo": rollout_stats["mean"] - ci95,
                "ci95_hi": rollout_stats["mean"] + ci95,
            }
            pd.DataFrame([summary_row]).to_csv(
                outdir / "rollout_summary.csv", index=False
            )
        else:
            pd.DataFrame(
                [{"n": 0, "mean_rollout_s": math.nan, "std_s": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan}]
            ).to_csv(outdir / "rollout_summary.csv", index=False)
    else:
        # Create placeholder figures
        plot_rollout_histograms(pd.DataFrame({"rollout_seconds": []}), outdir)
        pd.DataFrame(
            [{"n": 0, "mean_rollout_s": math.nan, "std_s": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan}]
        ).to_csv(outdir / "rollout_summary.csv", index=False)

    if args.decisions_csv:
        decisions_df = load_decisions_csv(Path(args.decisions_csv))
        decisions_df.to_csv(outdir / "decisions_local_copy.csv", index=False)

    latex = latex_table(headroom_summary, rollout_stats, decisions_summary)
    (outdir / "table_carla_metrics.tex").write_text(latex, encoding="utf-8")

    print(f"[OK] CARLA metrics written to {outdir}")


if __name__ == "__main__":
    main()
