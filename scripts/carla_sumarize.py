#!/usr/bin/env python3
import math, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ci95(mean, std, n):
    if n <= 1: return (mean, mean)
    m = 1.96 * (std / (n ** 0.5))
    return (mean - m, mean + m)

def main(rollouts_csv="data/carla_runs/rollouts_live.csv", outdir="data/carla_runs/summary_live"):
    rollouts_csv = Path(rollouts_csv)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(rollouts_csv)
    if "rollout_seconds" not in df.columns:
        # Try to auto-detect the column
        for c in df.columns:
            if "rollout" in c.lower():
                df = df.rename(columns={c:"rollout_seconds"})
                break
    assert "rollout_seconds" in df.columns, "Could not find rollout_seconds column."

    n = len(df)
    mu = df["rollout_seconds"].mean()
    sd = df["rollout_seconds"].std(ddof=1) if n > 1 else 0.0
    lo, hi = ci95(mu, sd, n)

    # Write summary CSV
    (outdir / "rollout_summary.csv").write_text(
        "n,mean_rollout_s,std_s,ci95_lo,ci95_hi\n" +
        f"{n},{mu:.3f},{sd:.3f},{lo:.3f},{hi:.3f}\n",
        encoding="utf-8"
    )

    # Histogram
    plt.figure()
    plt.hist(df["rollout_seconds"], bins=20)
    plt.xlabel("Rollout time (s)")
    plt.ylabel("Count")
    plt.title("CARLA live rollouts – histogram")
    plt.tight_layout()
    plt.savefig(outdir / "hist_rollouts.png")
    plt.close()

    # ECDF
    xs = sorted(df["rollout_seconds"].tolist())
    ys = [i / len(xs) for i in range(1, len(xs)+1)]
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Rollout time (s)")
    plt.ylabel("ECDF")
    plt.title("CARLA live rollouts – ECDF")
    plt.tight_layout()
    plt.savefig(outdir / "ecdf_rollouts.png")
    plt.close()

    # LaTeX table snippet
    latex = (
        "\\begin{table}[!t]\n"
        "  \\centering\n"
        "  \\caption{Live CARLA rollout timing summary: mean $\\pm \\sigma$ with 95\\% CI.}\n"
        "  \\label{tab:carla-live}\n"
        "  \\begin{tabular}{lrrrr}\n"
        "    \\toprule\n"
        "    Metric & $n$ & Mean (s) & Std (s) & 95\\% CI (s) \\\\\n"
        "    \\midrule\n"
        f"    Rollout total & {n} & {mu:.2f} & {sd:.2f} & [{lo:.2f}, {hi:.2f}] \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    (outdir / "table_carla_live.tex").write_text(latex, encoding="utf-8")

    print(f"[OK] Summary written to {outdir}/ (CSV, PNGs, and table_carla_live.tex)")

if __name__ == "__main__":
    rollouts = sys.argv[1] if len(sys.argv) > 1 else "data/carla_runs/rollouts_live.csv"
    outdir   = sys.argv[2] if len(sys.argv) > 2 else "data/carla_runs/summary_live"
    main(rollouts, outdir)
