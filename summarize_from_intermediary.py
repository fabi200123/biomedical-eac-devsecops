#!/usr/bin/env python3
import csv, math, statistics, os
from pathlib import Path

INPUT = "intermediary_metallb_argocd_timings.csv"
SUMMARY_MD = "intermediary_summary_table.md"
SUMMARY_TEX = "intermediary_summary_table.tex"
SUMMARY_CSV = "intermediary_summary_table.csv"
CHARTS_DIR = Path("charts")

# columns -> pretty names
METRICS = [
    ("repo_sync_s", "Repo sync"),
    ("manifest_apply_s", "Manifest apply"),
    ("resource_creation_s", "Resource creation"),
    ("deployment_total_s", "Deployment total"),
    ("pod_rollout_s", "Pod rollout"),
    ("rollout_total_s", "Rollout total"),
]

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def stats(values):
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), (float("nan"), float("nan")), 0)
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if n <= 1 else statistics.stdev(vals)
    if n > 1:
        se = sd / math.sqrt(n)
        ci = (mean - 1.96*se, mean + 1.96*se)
    else:
        ci = (float("nan"), float("nan"))
    return mean, sd, ci, n

def fmt(x, d=2):
    return "n/a" if x != x else f"{x:.{d}f}"  # NaN check via x!=x

def main():
    inp = Path(INPUT)
    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    # read rows
    with inp.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # only successful rows (file already filtered, but harmless)
    ok = [r for r in rows if r.get("status", "ok") == "ok"]
    # sort by iteration if present
    for r in ok:
        r["iteration"] = to_float(r.get("iteration"))
    ok = [r for r in ok if r["iteration"] is not None]
    ok.sort(key=lambda r: r["iteration"])

    # compute summary
    summary = []
    for col, label in METRICS:
        vals = [to_float(r.get(col)) for r in ok]
        mean, sd, (lo, hi), n = stats(vals)
        summary.append((label, mean, sd, lo, hi, n))

    # print summary
    print("\nCurrent summary (seconds):")
    print(f"{'Stage':<20} {'Mean ± σ':<18} {'95% CI':<22} {'n':>4}")
    print("-"*70)
    for label, mean, sd, lo, hi, n in summary:
        mean_sd = f"{fmt(mean)} ± {fmt(sd)}"
        ci = f"[{fmt(lo)}, {fmt(hi)}]" if n > 1 else "[n/a, n/a]"
        print(f"{label:<20} {mean_sd:<18} {ci:<22} {n:>4}")
    print("-"*70)

    # write markdown
    md = []
    md.append("**Argo CD deployment and rollout timing over current iterations (mean ± σ) with 95% confidence intervals.**\n")
    md.append("| Stage | Mean (s) ± σ | 95% CI (s) | n |")
    md.append("|---|---:|---:|---:|")
    for label, mean, sd, lo, hi, n in summary:
        mean_sd = f"{fmt(mean)} ± {fmt(sd)}"
        ci = f"[{fmt(lo)}, {fmt(hi)}]" if n > 1 else "[n/a, n/a]"
        md.append(f"| {label} | {mean_sd} | {ci} | {n} |")
    Path(SUMMARY_MD).write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {SUMMARY_MD}")

    # write LaTeX table (for your paper)
    tex_lines = [
        r"\begin{tabular}{l r r r}",
        r"\toprule",
        r"\textbf{Stage} & \textbf{Mean (s) $\pm \sigma$} & \textbf{95\% CI (s)} & \textbf{n} \\",
        r"\midrule",
    ]
    for label, mean, sd, lo, hi, n in summary:
        mean_sd = f"{fmt(mean)} $\\pm$ {fmt(sd)}"
        ci = f"[{fmt(lo)}, {fmt(hi)}]" if n > 1 else "[n/a, n/a]"
        tex_lines.append(f"{label} & {mean_sd} & {ci} & {n} \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    Path(SUMMARY_TEX).write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Wrote {SUMMARY_TEX}")

    # write CSV summary (handy for spreadsheets)
    with Path(SUMMARY_CSV).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage","mean_s","stddev_s","ci_low_s","ci_high_s","n"])
        for label, mean, sd, lo, hi, n in summary:
            w.writerow([label, fmt(mean,3), fmt(sd,3), fmt(lo,3), fmt(hi,3), n])
    print(f"Wrote {SUMMARY_CSV}")

    # charts (optional)
    try:
        import matplotlib.pyplot as plt  # only if present
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

        iters = [r["iteration"] for r in ok]
        for col, label in METRICS:
            series = [to_float(r.get(col)) for r in ok]
            pts = [(i, v) for i, v in zip(iters, series) if v is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)

            # line chart
            plt.figure()
            plt.plot(xs, ys, marker="o", linewidth=1)
            plt.title(f"{label} over iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Seconds")
            plt.tight_layout()
            line_path = CHARTS_DIR / f"{col}.png"
            plt.savefig(line_path, dpi=140)
            plt.close()

            # histogram
            plt.figure()
            plt.hist(ys, bins=min(30, max(5, int(len(ys) ** 0.5))))
            plt.title(f"{label} distribution")
            plt.xlabel("Seconds")
            plt.ylabel("Count")
            plt.tight_layout()
            hist_path = CHARTS_DIR / f"{col}_hist.png"
            plt.savefig(hist_path, dpi=140)
            plt.close()

            print(f"Saved charts: {line_path}, {hist_path}")

    except Exception as e:
        print(f"(Charts skipped: matplotlib not available or failed: {e})")

if __name__ == "__main__":
    main()
