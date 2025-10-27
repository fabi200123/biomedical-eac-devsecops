import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, tempfile, yaml, time
from pathlib import Path

# Import CARLA from the repo script
from scripts.carla_scheduler import Carla


def write_yaml(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")

def main():
    # Minimal policy that always allows (no clinic blocks), generous headroom
    policy = {
        "thresholds": {"cpu_watermark": 0.9, "ram_watermark": 0.9, "min_headroom": 0.0},
        "resources": {"Rmax": 3.0},
        "weights": {"alpha_depth": 1.0, "beta_invres": 1.0, "gamma_risk": 1.0, "delta_window": 1.0},
        "ema": {"rollout_alpha": 0.2},
        "windows": {"timezone": "Europe/Bucharest", "allow": [{"day": "WEEKDAY","start":"00:00","end":"23:59"}]},
        "human_in_the_loop": {"pause_high_risk": False},
        "tick": {"seconds": 1, "max_parallel_triggers": 1},
        "argo": {"base_url": "http://localhost:8080", "token_env": "ARGO_TOKEN", "verify_tls": False},
        "telemetry": {"prefer_k8s_metrics_api": False},
        "logging": {
            "decisions_csv": "data/carla_runs/decisions_offline.csv",
            "rollouts_csv": "data/carla_runs/rollouts_offline.csv",
            "level": "INFO",
        },
    }
    apps = {
        "apps": [
            {"name":"A_low","r_i":0.8,"d_i":2,"risk":0,"toggleRevisions":["revA","revB"]},
            {"name":"B_med","r_i":1.5,"d_i":1,"risk":1,"toggleRevisions":["revA","revB"]},
            {"name":"C_high","r_i":2.5,"d_i":0,"risk":2,"toggleRevisions":["revA","revB"]},
        ]
    }

    # Write temp configs
    ppol = Path("configs/_tmp_policy.yaml"); write_yaml(ppol, policy)
    papp = Path("configs/_tmp_apps.yaml");   write_yaml(papp, apps)

    # Fake token for ctor
    os.environ.setdefault("ARGO_TOKEN","DUMMY")

    # Monkeypatch Carla to avoid real Argo calls and sleep
    from scripts.carla_scheduler import ArgoClient, K8sMetrics
    ArgoClient.set_target_revision = lambda self, name, rev: None
    ArgoClient.wait_succeeded = lambda self, name, poll_s=1.0, timeout_s=1800: 0.2
    K8sMetrics.usage_fraction = lambda self: (0.2, 0.2)  # low usage => high headroom

    c = Carla(str(ppol), str(papp))
    # Run a few ticks
    for _ in range(3):
        c.run_once()
        time.sleep(0.2)

    print("OK: decisions at", policy["logging"]["decisions_csv"])
    print("OK: rollouts at", policy["logging"]["rollouts_csv"])

if __name__ == "__main__":
    main()
