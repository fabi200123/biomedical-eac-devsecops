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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def step_to_seconds(step: str) -> float:
    try:
        seconds = float(pd.to_timedelta(step).total_seconds())
        return seconds if seconds > 0 else 300.0
    except Exception:
        return 300.0


def integrate_rate_series(df: pd.DataFrame, step_seconds: float) -> float:
    if df.empty:
        return 0.0
    data = df.sort_values("ts")
    delta = data["ts"].shift(-1) - data["ts"]
    delta_seconds = delta.dt.total_seconds()
    fallback = delta_seconds.dropna().median()
    if math.isnan(fallback) or fallback <= 0:
        fallback = step_seconds
    delta_seconds = delta_seconds.fillna(fallback)
    return float((data["value"] * delta_seconds).sum())


def integrate_rate_by_label(df: pd.DataFrame, label_column: str, step_seconds: float) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    if df.empty or label_column not in df.columns:
        return totals
    for label, group in df.groupby(label_column):
        totals[str(label)] = integrate_rate_series(group[["ts", "value"]], step_seconds)
    return totals


def parse_bucket_bound(value: str) -> float:
    if value in ("+Inf", "Inf", "inf"):
        return math.inf
    return float(value)


def build_bucket_distribution(bucket_totals: Dict[str, float]) -> List[Dict[str, float]]:
    if not bucket_totals:
        return []
    entries = []
    for label, total in bucket_totals.items():
        try:
            upper = parse_bucket_bound(label)
        except ValueError:
            continue
        entries.append((upper, float(total)))
    entries.sort(key=lambda item: item[0])

    distribution: List[Dict[str, float]] = []
    prev_upper = 0.0
    prev_cum = 0.0
    prev_width: Optional[float] = None
    for upper, cumulative in entries:
        count = max(cumulative - prev_cum, 0.0)
        if math.isinf(upper):
            width = prev_width if prev_width and prev_width > 0 else (prev_upper if prev_upper > 0 else 1.0)
            bucket_upper = prev_upper + width
        else:
            bucket_upper = upper
            width = bucket_upper - prev_upper
            if width <= 0:
                width = prev_width if prev_width and prev_width > 0 else 1.0
                bucket_upper = prev_upper + width
            prev_width = width
        distribution.append({"lower": prev_upper, "upper": bucket_upper, "count": count})
        prev_upper = bucket_upper
        prev_cum = cumulative
    return distribution


def distribution_quantile(distribution: List[Dict[str, float]], total: float, quantile: float) -> float:
    if not distribution or total <= 0:
        return math.nan
    target = max(min(quantile, 1.0), 0.0) * total
    acc = 0.0
    for bucket in distribution:
        count = bucket["count"]
        if count <= 0:
            continue
        next_acc = acc + count
        if target <= next_acc:
            width = bucket["upper"] - bucket["lower"]
            if width <= 0:
                return bucket["upper"]
            fraction = (target - acc) / count if count > 0 else 0.0
            return bucket["lower"] + fraction * width
        acc = next_acc
    return distribution[-1]["upper"]


def bucket_distribution_stats(distribution: List[Dict[str, float]]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if not distribution:
        return stats
    total = sum(bucket["count"] for bucket in distribution)
    if total <= 0:
        return stats
    mean = sum(bucket["count"] * 0.5 * (bucket["lower"] + bucket["upper"]) for bucket in distribution) / total
    second_moment = (
        sum(
            bucket["count"] * (bucket["lower"] ** 2 + bucket["lower"] * bucket["upper"] + bucket["upper"] ** 2) / 3.0
            for bucket in distribution
        )
        / total
    )
    variance = max(second_moment - mean**2, 0.0)
    stats["mean"] = mean
    stats["std"] = math.sqrt(variance)
    stats["median"] = distribution_quantile(distribution, total, 0.5)
    stats["p90"] = distribution_quantile(distribution, total, 0.9)
    stats["p95"] = distribution_quantile(distribution, total, 0.95)
    return stats


def compute_rollout_stats_from_prom(
    count_df: pd.DataFrame,
    sum_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    step_seconds: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    if count_df.empty and sum_df.empty and bucket_df.empty:
        return {}, []
    total_rollouts = integrate_rate_series(count_df[["ts", "value"]], step_seconds) if not count_df.empty else 0.0
    total_duration = integrate_rate_series(sum_df[["ts", "value"]], step_seconds) if not sum_df.empty else math.nan
    bucket_totals = integrate_rate_by_label(bucket_df, "label_le", step_seconds)
    distribution = build_bucket_distribution(bucket_totals)
    distribution_total = sum(bucket["count"] for bucket in distribution) if distribution else 0.0
    effective_n = total_rollouts if total_rollouts > 0 else distribution_total
    if effective_n <= 0:
        return {}, distribution
    stats: Dict[str, float] = {"n": effective_n}
    dist_stats = bucket_distribution_stats(distribution) if distribution else {}
    mean_from_sum = (
        total_duration / effective_n if effective_n > 0 and not math.isnan(total_duration) else math.nan
    )
    if not math.isnan(mean_from_sum):
        stats["mean"] = mean_from_sum
    elif dist_stats:
        stats["mean"] = dist_stats.get("mean", math.nan)
    else:
        stats["mean"] = math.nan
    stats["std"] = dist_stats.get("std", math.nan) if dist_stats else math.nan
    stats["median"] = dist_stats.get("median", math.nan) if dist_stats else math.nan
    stats["p90"] = dist_stats.get("p90", math.nan) if dist_stats else math.nan
    stats["p95"] = dist_stats.get("p95", math.nan) if dist_stats else math.nan
    return stats, distribution


def synthesize_rollout_samples(distribution: List[Dict[str, float]], max_samples: int = 2000) -> pd.DataFrame:
    if not distribution:
        return pd.DataFrame({"rollout_seconds": []})
    total = sum(bucket["count"] for bucket in distribution)
    if total <= 0:
        return pd.DataFrame({"rollout_seconds": []})
    scaling = total / max_samples if total > max_samples else 1.0
    samples: List[float] = []
    for bucket in distribution:
        count = bucket["count"]
        lower = bucket["lower"]
        upper = bucket["upper"]
        if count <= 0:
            continue
        if upper <= lower:
            samples.append(lower)
            continue
        if scaling > 1.0:
            sample_n = int(max(1, round(count / scaling)))
        else:
            sample_n = int(max(1, round(count)))
        for i in range(sample_n):
            frac = (i + 0.5) / sample_n
            samples.append(lower + frac * (upper - lower))
    return pd.DataFrame({"rollout_seconds": samples})


def write_window_summary(
    out_path: Path,
    start: Optional[str],
    end: Optional[str],
    rollout_stats: Dict[str, float],
) -> None:
    rows: List[Dict[str, Any]] = []
    start_ts = pd.to_datetime(start) if start else None
    end_ts = pd.to_datetime(end) if end else None
    if start_ts is not None:
        rows.append({"metric": "window_start", "value": start_ts.isoformat()})
    if end_ts is not None:
        rows.append({"metric": "window_end", "value": end_ts.isoformat()})
    if start_ts is not None and end_ts is not None:
        window_seconds = max((end_ts - start_ts).total_seconds(), 0.0)
        rows.append({"metric": "window_hours", "value": window_seconds / 3600.0})
    if rollout_stats:
        rows.append({"metric": "prom_rollouts_total", "value": rollout_stats.get("n", math.nan)})
        rows.append({"metric": "prom_rollout_mean_s", "value": rollout_stats.get("mean", math.nan)})
        rows.append({"metric": "prom_rollout_std_s", "value": rollout_stats.get("std", math.nan)})
        rows.append({"metric": "prom_rollout_p90_s", "value": rollout_stats.get("p90", math.nan)})
        rows.append({"metric": "prom_rollout_p95_s", "value": rollout_stats.get("p95", math.nan)})
    if not rows:
        rows = [{"metric": "info", "value": "No Prometheus metadata available"}]
    pd.DataFrame(rows).to_csv(out_path, index=False)


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
) -> Dict[str, Any]:
    """Collect required Prometheus metrics and write raw CSV files."""
    results: Dict[str, pd.DataFrame] = {}
    step_seconds = max(step_to_seconds(step), 1.0)

    queries = {
        "headroom": "carla_headroom_ratio",
        "decisions_rate": "sum(rate(carla_decisions_total[5m])) by (status)",
        "rollout_mean": (
            "sum(rate(carla_rollout_duration_seconds_sum[5m])) "
            "/ sum(rate(carla_rollout_duration_seconds_count[5m]))"
        ),
        "rollout_sum_rate": "sum(rate(carla_rollout_duration_seconds_sum[5m]))",
        "rollout_count_rate": "sum(rate(carla_rollout_duration_seconds_count[5m]))",
        "rollout_buckets": "sum(rate(carla_rollout_duration_seconds_bucket[5m])) by (le)",
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

    rollout_stats_prom, distribution = compute_rollout_stats_from_prom(
        results.get("rollout_count_rate", pd.DataFrame()),
        results.get("rollout_sum_rate", pd.DataFrame()),
        results.get("rollout_buckets", pd.DataFrame()),
        step_seconds,
    )
    results["rollout_stats_prom"] = rollout_stats_prom
    results["rollout_distribution_prom"] = distribution

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


def plot_rollout_quantiles_timeseries(
    quantiles_df: pd.DataFrame,
    out_path: Path,
    timezone: str,
    global_stats: Optional[Dict[str, float]] = None,
) -> None:
    plt.figure(figsize=(10, 4))
    if quantiles_df.empty:
        plt.text(0.5, 0.5, "No rollout quantiles available", ha="center", va="center")
        plt.axis("off")
    else:
        data = quantiles_df.copy().sort_values("ts")
        if "ts" not in data.columns:
            plt.text(0.5, 0.5, "Quantiles missing timestamps", ha="center", va="center")
            plt.axis("off")
        else:
            data = data.dropna(subset=["ts"])
            data["ts"] = pd.to_datetime(data["ts"], utc=True)
            data["ts_local"] = data["ts"].dt.tz_convert(timezone)
            numeric = data[["p50", "p90", "p95"]].astype(float)
            numeric = numeric.interpolate(limit_direction="both")
            valid = numeric.dropna(how="all")
            if valid.empty:
                plt.text(0.5, 0.5, "Quantile series empty", ha="center", va="center")
                plt.axis("off")
            else:
                data[["p50", "p90", "p95"]] = numeric
                plt.plot(data["ts_local"], data["p50"], label="p50", color="#1f77b4")
                plt.fill_between(
                    data["ts_local"],
                    data["p50"],
                    data["p90"],
                    color="#1f77b4",
                    alpha=0.2,
                    label="p50–p90",
                )
                plt.fill_between(
                    data["ts_local"],
                    data["p90"],
                    data["p95"],
                    color="#ff7f0e",
                    alpha=0.2,
                    label="p90–p95",
                )
                plt.plot(data["ts_local"], data["p90"], label="p90", color="#ff7f0e", linewidth=1.5)
                plt.plot(data["ts_local"], data["p95"], label="p95", color="#d62728", linewidth=1.5)
                if global_stats:
                    for level, style, color in [
                        ("p90", "--", "#ff7f0e"),
                        ("p95", ":", "#d62728"),
                    ]:
                        val = global_stats.get(level)
                        if val is None or math.isnan(val):
                            continue
                        plt.axhline(val, linestyle=style, color=color, alpha=0.8)
                        plt.text(
                            data["ts_local"].min(),
                            val,
                            f"global {level} {val:.2f}s",
                            color=color,
                            va="bottom",
                            fontsize=9,
                        )
                plt.title("CARLA Rollout Duration Quantiles")
                plt.xlabel(f"Time ({timezone})")
                plt.ylabel("Seconds")
                plt.grid(alpha=0.2)
                plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rollout_histograms(
    rollouts_df: pd.DataFrame,
    outdir: Path,
    annotation_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
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
    stats = {
        "n": len(xs),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
        "p95": float(values.quantile(0.95)),
    }
    overlay_stats = annotation_stats or stats

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20)
    plt.title("Rollout Durations")
    plt.xlabel("Seconds")
    plt.ylabel("Count")
    if overlay_stats:
        y_max = plt.gca().get_ylim()[1]
        for level, color, linestyle in [
            ("p90", "#ff7f0e", "--"),
            ("p95", "#d62728", ":"),
        ]:
            marker = overlay_stats.get(level)
            if marker is None or math.isnan(marker):
                continue
            plt.axvline(marker, color=color, linestyle=linestyle, linewidth=1.4)
            plt.text(
                marker,
                y_max * 0.9,
                f"{level.upper()} {marker:.2f}s",
                rotation=90,
                color=color,
                va="top",
                ha="right",
                fontsize=8,
            )
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
    if overlay_stats:
        for level, color, linestyle in [
            ("p90", "#ff7f0e", "--"),
            ("p95", "#d62728", ":"),
        ]:
            marker = overlay_stats.get(level)
            if marker is None or math.isnan(marker):
                continue
            plt.axvline(marker, color=color, linestyle=linestyle, linewidth=1.4)
            plt.text(
                marker,
                0.1 if level == "p90" else 0.3,
                f"{level.upper()} {marker:.2f}s",
                rotation=90,
                color=color,
                va="bottom",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(outdir / "fig_rollout_ecdf.png")
    plt.close()

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

    prom_results: Dict[str, Any] = {}
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

    prom_rollout_stats = prom_results.get("rollout_stats_prom", {}) if prom_results else {}
    prom_rollout_distribution = prom_results.get("rollout_distribution_prom", []) if prom_results else []
    rollout_stats: Dict[str, float] = dict(prom_rollout_stats) if prom_rollout_stats else {}
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
        if rollout_stats:
            n_float = max(float(rollout_stats.get("n", 0.0)), 0.0)
            n = max(int(round(n_float)), 0)
            std_val = rollout_stats.get("std", math.nan)
            mean_val = rollout_stats.get("mean", math.nan)
            ci95 = 1.96 * (std_val / math.sqrt(n_float)) if n_float > 1 and not math.isnan(std_val) else 0.0
            summary_row = {
                "n": n,
                "mean_rollout_s": mean_val,
                "std_s": std_val,
                "ci95_lo": mean_val - ci95 if not math.isnan(mean_val) else math.nan,
                "ci95_hi": mean_val + ci95 if not math.isnan(mean_val) else math.nan,
            }
            pd.DataFrame([summary_row]).to_csv(outdir / "rollout_summary.csv", index=False)
            synthetic_rollouts = synthesize_rollout_samples(prom_rollout_distribution)
            plot_rollout_histograms(synthetic_rollouts, outdir, annotation_stats=rollout_stats)
        else:
            # Create placeholder figures
            plot_rollout_histograms(pd.DataFrame({"rollout_seconds": []}), outdir)
            pd.DataFrame(
                [{"n": 0, "mean_rollout_s": math.nan, "std_s": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan}]
            ).to_csv(outdir / "rollout_summary.csv", index=False)

    if args.decisions_csv:
        decisions_df = load_decisions_csv(Path(args.decisions_csv))
        decisions_df.to_csv(outdir / "decisions_local_copy.csv", index=False)

    plot_rollout_quantiles_timeseries(
        rollout_quantiles,
        outdir / "fig_rollout_quantiles.png",
        args.timezone,
        rollout_stats if rollout_stats else prom_rollout_stats,
    )
    write_window_summary(
        outdir / "summary_window.csv",
        args.start,
        args.end,
        prom_rollout_stats if prom_rollout_stats else rollout_stats,
    )

    latex = latex_table(headroom_summary, rollout_stats, decisions_summary)
    (outdir / "table_carla_metrics.tex").write_text(latex, encoding="utf-8")

    print(f"[OK] CARLA metrics written to {outdir}")


if __name__ == "__main__":
    main()
