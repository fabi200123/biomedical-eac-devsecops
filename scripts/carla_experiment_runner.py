"""
CARLA Experiment Runner
-----------------------
Offline, reproducible evaluation of rollout strategies:
  - FIFO (baseline)
  - WAVE (time-based cadence)
  - CARLA-Greedy (priority-only)
  - CARLA-Knapsack (priority + resource fit)

Outputs:
  data/experiments/<run_id>/
    decisions_<strategy>.csv
    rollouts_<strategy>.csv
    summary_<strategy>.csv
    (optional) plots/*.png

If present, will sample rollout durations from:
  data/carla_runs/rollouts_offline.csv
else defaults to N(21.8s, 5.5s) truncated to [5, 90].
"""

import os, sys, math, time, json, random
import datetime as dt
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing dependency: pyyaml. pip install pyyaml")
    sys.exit(1)

# Optional libs for richer summaries/plots
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

# Matplotlib is optional – plots are only generated if available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ------------------------------ Helpers ------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path, default=None):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return default


def truncated_normal(mu=21.8, sigma=5.52, lo=5.0, hi=90.0):
    """Draws a rollout time sample from a truncated normal."""
    # Fallback if numpy not available
    if np is None:
        x = random.gauss(mu, sigma)
        return max(lo, min(hi, x))
    x = np.random.normal(mu, sigma)
    return float(np.clip(x, lo, hi))


def sample_rollout_seconds(rollouts_csv: Path | None):
    """
    Returns a callable that samples rollout total (seconds).
    If rollouts_csv exists and has 'Rollout total' column, sample from it.
    """
    if rollouts_csv and rollouts_csv.exists() and pd is not None:
        try:
            df = pd.read_csv(rollouts_csv)
            col = None
            for c in df.columns:
                if "Rollout total" in c or "rollout" in c.lower():
                    col = c; break
            if col is not None and len(df[col].dropna()) >= 20:
                vals = df[col].dropna().astype(float).values
                def s():
                    return float(random.choice(vals))
                return s
        except Exception:
            pass
    # default parametric sampler
    return lambda: truncated_normal()


# ------------------------------ Synthetic cluster model ------------------------------

class ClusterState:
    """
    Simple synthetic model of cluster headroom.
    We model CPU and RAM as fractions in [0,1] used (not headroom);
    policy defines max allowed utilization after scheduling.
    """
    def __init__(self, cpu_used=0.25, ram_used=0.24,
                 cpu_sigma=0.05, ram_sigma=0.06, seed=42):
        self.rng = random.Random(seed)
        self.cpu = cpu_used
        self.ram = ram_used
        self.cpu_sigma = cpu_sigma
        self.ram_sigma = ram_sigma

    def tick(self):
        # Random walk with light noise, clamp to [0, 0.95]
        self.cpu = max(0.0, min(0.95, self.cpu + self.rng.uniform(-self.cpu_sigma, self.cpu_sigma)))
        self.ram = max(0.0, min(0.95, self.ram + self.rng.uniform(-self.ram_sigma, self.ram_sigma)))

    def would_fit(self, cpu_delta, ram_delta, max_cpu_after, max_ram_after):
        return (self.cpu + cpu_delta) <= max_cpu_after and (self.ram + ram_delta) <= max_ram_after

    def apply(self, cpu_delta, ram_delta):
        self.cpu = min(0.999, self.cpu + cpu_delta)
        self.ram = min(0.999, self.ram + ram_delta)


# ------------------------------ App model ------------------------------

def load_apps(apps_yaml: Path | None):
    """
    Load apps with fields:
      name, priority (float), cpu_cost (fraction), ram_cost (fraction), group (optional)
    If configs/apps.yaml absent, return a small synthetic set.
    """
    default = [
        {"name": "mgmt/metallb", "priority": 0.90, "cpu_cost": 0.04, "ram_cost": 0.03},
        {"name": "apps/onco-ct", "priority": 0.85, "cpu_cost": 0.07, "ram_cost": 0.06},
        {"name": "apps/onco-ocr", "priority": 0.80, "cpu_cost": 0.05, "ram_cost": 0.05},
        {"name": "apps/onco-api", "priority": 0.70, "cpu_cost": 0.03, "ram_cost": 0.03},
        {"name": "apps/onco-front", "priority": 0.60, "cpu_cost": 0.02, "ram_cost": 0.02},
        {"name": "apps/etl", "priority": 0.50, "cpu_cost": 0.06, "ram_cost": 0.05},
    ]
    if not apps_yaml or not apps_yaml.exists():
        return default
    y = load_yaml(apps_yaml)
    if isinstance(y, list):
        return y
    if isinstance(y, dict) and "apps" in y:
        return y["apps"]
    return default


def load_policy(policy_yaml: Path | None):
    """
    Policy fields (with defaults):
      max_cpu_after: 0.7
      max_ram_after: 0.7
      max_apps_per_tick: 3
      wave_period: 10   (ticks)
      wave_phase: 0
      risk_boost: 0.0 .. 1.0  (additive boost to priority)
      tick_seconds: 30
      seed: 123
    """
    defaults = {
        "max_cpu_after": 0.70,
        "max_ram_after": 0.70,
        "max_apps_per_tick": 3,
        "wave_period": 10,
        "wave_phase": 0,
        "risk_boost": 0.0,
        "tick_seconds": 30,
        "seed": 123
    }
    if policy_yaml and policy_yaml.exists():
        data = load_yaml(policy_yaml, {})
        defaults.update({k: v for k, v in (data or {}).items() if k in defaults})
    return defaults


# ------------------------------ Strategies ------------------------------

class StrategyBase:
    def __init__(self, name):
        self.name = name

    def select(self, apps, cluster: ClusterState, policy, t):
        """
        Return a (possibly empty) list of apps to rollout at tick t.
        """
        raise NotImplementedError


class FifoStrategy(StrategyBase):
    def __init__(self): super().__init__("fifo")
    def select(self, apps, cluster, policy, t):
        chosen = []
        for app in apps:
            if len(chosen) >= policy["max_apps_per_tick"]:
                break
            if cluster.would_fit(app["cpu_cost"], app["ram_cost"], policy["max_cpu_after"], policy["max_ram_after"]):
                chosen.append(app)
                # Reserve locally to not oversubscribe in this tick
                cluster.apply(app["cpu_cost"], app["ram_cost"])
        return chosen


class WaveStrategy(StrategyBase):
    def __init__(self): super().__init__("wave")
    def select(self, apps, cluster, policy, t):
        period = max(1, int(policy["wave_period"]))
        phase = int(policy["wave_phase"])
        if ((t + phase) % period) != 0:
            return []
        # same as FIFO when wave is "on"
        return FifoStrategy().select(apps, cluster, policy, t)


class CarlaGreedy(StrategyBase):
    def __init__(self): super().__init__("carla_greedy")
    def select(self, apps, cluster, policy, t):
        # Compute score = priority + risk_boost (if any)
        scored = []
        for a in apps:
            score = float(a.get("priority", 0.5)) + float(policy.get("risk_boost", 0.0))
            scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)

        chosen = []
        for _, app in scored:
            if len(chosen) >= policy["max_apps_per_tick"]:
                break
            if cluster.would_fit(app["cpu_cost"], app["ram_cost"], policy["max_cpu_after"], policy["max_ram_after"]):
                chosen.append(app)
                cluster.apply(app["cpu_cost"], app["ram_cost"])
        return chosen


class CarlaKnapsack(StrategyBase):
    def __init__(self): super().__init__("carla_knapsack")
    def select(self, apps, cluster, policy, t):
        """
        Greedy 2D knapsack heuristic:
          value = priority
          weights = (cpu_cost, ram_cost)
          capacity = (max_cpu_after - cpu_used, max_ram_after - ram_used)
        """
        cap_cpu = max(0.0, policy["max_cpu_after"] - cluster.cpu)
        cap_ram = max(0.0, policy["max_ram_after"] - cluster.ram)
        if cap_cpu <= 0.0 or cap_ram <= 0.0:
            return []

        scored = []
        for a in apps:
            v = float(a.get("priority", 0.5)) + float(policy.get("risk_boost", 0.0))
            w_cpu = a["cpu_cost"]; w_ram = a["ram_cost"]
            denom = (w_cpu + w_ram) or 1e-6
            density = v / denom
            scored.append((density, a))
        scored.sort(key=lambda x: x[0], reverse=True)

        chosen = []
        used_cpu = 0.0; used_ram = 0.0
        for _, app in scored:
            if len(chosen) >= policy["max_apps_per_tick"]:
                break
            if (used_cpu + app["cpu_cost"] <= cap_cpu) and (used_ram + app["ram_cost"] <= cap_ram):
                chosen.append(app)
                used_cpu += app["cpu_cost"]
                used_ram += app["ram_cost"]

        for app in chosen:
            cluster.apply(app["cpu_cost"], app["ram_cost"])
        return chosen


# ------------------------------ Experiment ------------------------------

def run_strategy(run_id: str,
                 strategy: StrategyBase,
                 apps: list,
                 policy: dict,
                 ticks: int = 60,
                 initial_cpu=0.25,
                 initial_ram=0.24,
                 sampler=None,
                 out_root=Path("data/experiments")):

    if sampler is None:
        sampler = sample_rollout_seconds(Path("data/carla_runs/rollouts_offline.csv"))

    rng = random.Random(policy.get("seed", 123))
    run_dir = out_root / run_id
    ensure_dir(run_dir)
    decisions_path = run_dir / f"decisions_{strategy.name}.csv"
    rollouts_path = run_dir / f"rollouts_{strategy.name}.csv"
    summary_path = run_dir / f"summary_{strategy.name}.csv"
    plots_dir = run_dir / "plots"
    ensure_dir(plots_dir)

    # CSV headers
    with decisions_path.open("w", encoding="utf-8") as f:
        f.write("tick,timestamp,strategy,app,priority,cpu_cost,ram_cost,cpu_used_after,ram_used_after\n")
    with rollouts_path.open("w", encoding="utf-8") as f:
        f.write("tick,strategy,app,rollout_seconds\n")

    cluster = ClusterState(cpu_used=initial_cpu, ram_used=initial_ram, seed=policy.get("seed", 123))

    for t in range(ticks):
        cluster.tick()
        now = dt.datetime.utcnow().isoformat()

        # copy cluster for selection phase (so selection sees the headroom at tick t)
        sel_cluster = ClusterState(cluster.cpu, cluster.ram, 0.0, 0.0, seed=policy.get("seed", 123))
        chosen = strategy.select(apps, sel_cluster, policy, t)

        # apply chosen to the "real" cluster and log decisions
        for app in chosen:
            cluster.apply(app["cpu_cost"], app["ram_cost"])
            with decisions_path.open("a", encoding="utf-8") as f:
                f.write(f"{t},{now},{strategy.name},{app['name']},{app.get('priority',0.5):.3f},"
                        f"{app['cpu_cost']:.3f},{app['ram_cost']:.3f},{cluster.cpu:.3f},{cluster.ram:.3f}\n")

            # rollout time sample
            s = sampler()
            with rollouts_path.open("a", encoding="utf-8") as f:
                f.write(f"{t},{strategy.name},{app['name']},{s:.3f}\n")

    # Summary
    if pd is not None and (rollouts_path.exists()):
        df = pd.read_csv(rollouts_path)
        if len(df) > 0:
            mu = df["rollout_seconds"].mean()
            sd = df["rollout_seconds"].std(ddof=1) if len(df) > 1 else 0.0
            n = len(df)
            ci_lo = mu - 1.96 * (sd / math.sqrt(n)) if n > 1 else mu
            ci_hi = mu + 1.96 * (sd / math.sqrt(n)) if n > 1 else mu
        else:
            mu = sd = 0.0; n = 0; ci_lo = ci_hi = 0.0

        with summary_path.open("w", encoding="utf-8") as f:
            f.write("strategy,n,mean_rollout_s,std_s,ci95_lo,ci95_hi\n")
            f.write(f"{strategy.name},{n},{mu:.3f},{sd:.3f},{ci_lo:.3f},{ci_hi:.3f}\n")

        # Plots (if matplotlib available)
        if plt is not None and n > 0:
            # Histogram
            plt.figure()
            plt.hist(df["rollout_seconds"], bins=20)
            plt.xlabel("Rollout time (s)")
            plt.ylabel("Count")
            plt.title(f"Histogram: {strategy.name}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"hist_{strategy.name}.png")
            plt.close()

            # ECDF
            xs = sorted(df["rollout_seconds"].tolist())
            ys = [i / len(xs) for i in range(1, len(xs) + 1)]
            plt.figure()
            plt.plot(xs, ys)
            plt.xlabel("Rollout time (s)")
            plt.ylabel("ECDF")
            plt.title(f"ECDF: {strategy.name}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"ecdf_{strategy.name}.png")
            plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run offline CARLA scheduling experiments.")
    ap.add_argument("--run-id", default=dt.datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ"))
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--apps", default="configs/apps.yaml")
    ap.add_argument("--policy", default="configs/carla_policy.yaml")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    apps = load_apps(Path(args.apps))
    policy = load_policy(Path(args.policy))
    policy["seed"] = args.seed

    strategies = [
        FifoStrategy(),
        WaveStrategy(),
        CarlaGreedy(),
        CarlaKnapsack(),
    ]

    for s in strategies:
        print(f"[INFO] Running strategy: {s.name}")
        run_strategy(run_id=args.run_id,
                     strategy=s,
                     apps=apps,
                     policy=policy,
                     ticks=args.ticks)

    print(f"[OK] Results at data/experiments/{args.run_id}/")


if __name__ == "__main__":
    main()
