import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, yaml, time
from pathlib import Path
from scripts.carla_scheduler import Carla, ArgoClient, K8sMetrics


def main():
    # Use your real configs if you want; here we re-use the ones from Step 1
    policy_path = "configs/carla_policy.yaml"
    apps_path   = "configs/apps.yaml"
    assert Path(policy_path).exists() and Path(apps_path).exists()

    # Provide a dummy token
    os.environ.setdefault("ARGO_TOKEN","DUMMY")

    # Monkeypatch Argo + Metrics
    def fake_get_app(self, name):
        return {"metadata":{"name":name},"spec":{"source":{"targetRevision":"revA"}},
                "status":{"operationState":{"phase":"Succeeded"}}}
    def fake_set_rev(self, name, rev):
        print(f"[DRYRUN] set_target_revision({name}, {rev})")
    def fake_wait(self, name, poll_s=1.0, timeout_s=1800):
        # emulate ~0.25s rollout
        time.sleep(0.05); return 0.25
    def fake_usage(self):
        # simulate moderate load
        return (0.45, 0.50)

    ArgoClient.get_app = fake_get_app
    ArgoClient.set_target_revision = fake_set_rev
    ArgoClient.wait_succeeded = fake_wait
    K8sMetrics.usage_fraction = fake_usage

    c = Carla(policy_path, apps_path)
    c.run_once()
    print("OK: one scheduler tick completed (dry run).")

if __name__ == "__main__":
    main()
