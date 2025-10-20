#!/usr/bin/env python3
import os
import sys
import traceback
import pandas as pd
import matplotlib.pyplot as plt

print("=== Absolute-path rollout plot generator (safe) ===")

# Locate repo root and inputs
repo_root = os.path.dirname(os.path.abspath(__file__))
print("Repo root:", repo_root)

inputs = [
    os.path.join(repo_root, "../data/rollout_summaries/metallb_argocd_timings.csv"),
    os.path.join(repo_root, "../data/rollout_summaries/metallb_argocd_timings_2.csv"),
]

# Prefer project subfolder "_out_fig" (not "fig" to avoid conflicts).
# If that fails, fall back to user's Documents.
primary_out = os.path.join(repo_root, "_out_fig")
fallback_out = os.path.expanduser("~/Documents/biomedical_figs")
out_dir = None

def ensure_out_dir(path):
    print("[INFO] Ensuring output dir:", path)
    # If a FILE exists with this name, rename it aside
    if os.path.isfile(path):
        backup = path + "_old"
        print(f"[WARN] Blocking FILE found at {path}. Renaming to: {backup}")
        os.replace(path, backup)
    try:
        os.makedirs(path, exist_ok=True)
        print("[OK] Output dir ready:", path)
        return path
    except Exception as e:
        print(f"[ERROR] Could not create {path}: {e}")
        return None

# Try primary, then fallback
out_dir = ensure_out_dir(primary_out) or ensure_out_dir(fallback_out)
if not out_dir:
    print("[FATAL] Could not create any output directory. Exiting.")
    sys.exit(2)

# Read inputs
dfs = []
for p in inputs:
    print(f"[INFO] Reading: {p} exists? {os.path.exists(p)}")
    if not os.path.exists(p):
        print(f"[ERROR] Missing input CSV: {p}")
        sys.exit(3)
    try:
        df = pd.read_csv(p)
        print(f"[OK] Loaded {len(df)} rows; cols: {list(df.columns)}")
        dfs.append(df)
    except Exception as e:
        print(f"[ERROR] Failed to read {p}: {e}")
        traceback.print_exc()
        sys.exit(4)

# Combine + filter
df = pd.concat(dfs, ignore_index=True)
if "status" not in df.columns or "rollout_total_s" not in df.columns:
    print("[ERROR] Required columns not found. Columns present:", list(df.columns))
    sys.exit(5)

ok = df[df["status"] == "ok"].copy()
print("[INFO] Successful rows (status=ok):", len(ok))

# Convert types
ok["rollout_total_s"] = pd.to_numeric(ok["rollout_total_s"], errors="coerce")
ok["iteration_global"] = range(1, len(ok) + 1)

# --- HISTOGRAM ---
try:
    print("[INFO] Generating histogram ...")
    plt.figure()
    ok["rollout_total_s"].plot(kind="hist", bins=30, edgecolor="black")
    plt.xlabel("Rollout total (s)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of rollout totals (n={len(ok)})")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "fig_rollout_hist.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print("[OK] Wrote:", hist_path, "exists?", os.path.exists(hist_path))
except Exception as e:
    print("[ERROR] Histogram failed:", e)
    traceback.print_exc()

# --- SERIES ---
try:
    print("[INFO] Generating time series ...")
    plt.figure()
    plt.plot(ok["iteration_global"], ok["rollout_total_s"])
    plt.xlabel("Successful iteration (combined runs)")
    plt.ylabel("Rollout total (s)")
    plt.title(f"Rollout total over iterations (n={len(ok)})")
    plt.tight_layout()
    series_path = os.path.join(out_dir, "fig_rollout_series.png")
    plt.savefig(series_path, dpi=200)
    plt.close()
    print("[OK] Wrote:", series_path, "exists?", os.path.exists(series_path))
except Exception as e:
    print("[ERROR] Series plot failed:", e)
    traceback.print_exc()

# Final listing
try:
    listing = os.listdir(out_dir)
    print("[INFO] Final output listing:", listing)
except Exception as e:
    print("[ERROR] Could not list output dir:", e)

print("=== Done ===")
