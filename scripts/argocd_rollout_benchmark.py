#!/usr/bin/env python3
import csv
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone

# --- CONFIG ---
APP_NAME = "metallb"
APP_NAMESPACE = "mgmt"                 # Argo CD Application CR namespace
TARGET_REVISIONS = ["v0.15.1","v0.15.2"]  # revisions to toggle
ITERATIONS = 1000
CSV_PATH = "../data/rollout_summaries/metallb_argocd_timings.csv"
K8S_TARGET_NS = "metallb-system"       # where the chart installs resources
# ---------------

ARGOCD_NS_FLAGS = ["--app-namespace", APP_NAMESPACE]

# ---- helpers ----
def sh(cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)

def iso_utc(ts):
    if ts is None: return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")

def iso_local(ts):
    if ts is None: return ""
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")

def fmt2(x):
    return "" if x is None else f"{x:.3f}"

def argocd_app_json():
    out = sh(["argocd","app","get",APP_NAME,"-o","json",*ARGOCD_NS_FLAGS]).stdout
    return json.loads(out)

def argocd_sync(timeout=600):
    sh(["argocd","app","sync",APP_NAME,"--timeout",str(timeout),*ARGOCD_NS_FLAGS])

def argocd_wait_healthy(timeout=600):
    sh(["argocd","app","wait",APP_NAME,"--health","--timeout",str(timeout),*ARGOCD_NS_FLAGS])

def kubectl_patch_revision(rev):
    patch = {"spec":{"source":{"targetRevision":rev}}}
    sh(["kubectl","-n",APP_NAMESPACE,"patch","application",APP_NAME,"--type=merge","-p",json.dumps(patch)])

def get_target_revision(appj):
    return appj.get("spec",{}).get("source",{}).get("targetRevision")

def get_op_times(appj):
    op = (appj.get("status") or {}).get("operationState")
    if not op: return (None,None)
    def parse(ts):
        if not ts: return None
        return datetime.strptime(ts,"%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    return (parse(op.get("startedAt")), parse(op.get("finishedAt")))

def list_sync_resources(appj):
    op = (appj.get("status") or {}).get("operationState") or {}
    sync = op.get("syncResult") or {}
    return sync.get("resources") or []

def all_synced(resources):
    # Argo CD sets resource.status to "Synced" when apply finished for that resource
    return len(resources) > 0 and all((r.get("status") == "Synced") for r in resources)

def wait_until(fn, timeout_s, interval_s=0.5):
    t0 = time.time()
    while True:
        ok, payload = fn()
        if ok: return (True, payload, time.time())
        if time.time() - t0 > timeout_s:
            return (False, payload, time.time())
        time.sleep(interval_s)

# ---- rollout checks for workloads in target namespace ----
def k_get(kind, extra_args=None):
    extra_args = extra_args or []
    out = sh(["kubectl","-n",K8S_TARGET_NS,"get",kind,"-o","json",*extra_args]).stdout
    return json.loads(out)

def workloads_ready():
    try:
        ds = k_get("daemonsets")
        sts = k_get("statefulsets")
        dep = k_get("deployments")
    except subprocess.CalledProcessError as e:
        return False, f"k8s get error: {e.stderr.strip()}"

    # deployments: condition Available true and updatedReplicas==replicas
    for d in dep.get("items",[]):
        spec_repl = (d.get("spec") or {}).get("replicas",0) or 0
        status = d.get("status") or {}
        if status.get("observedGeneration",0) < d.get("metadata",{}).get("generation",0):
            return False, "deployment generation not observed"
        if (status.get("availableReplicas") or 0) < spec_repl:
            return False, "deployment not available"
        if (status.get("updatedReplicas") or 0) < spec_repl:
            return False, "deployment not fully updated"

    # statefulsets: readyReplicas==replicas
    for s in sts.get("items",[]):
        spec_repl = (s.get("spec") or {}).get("replicas",0) or 0
        status = s.get("status") or {}
        if (status.get("readyReplicas") or 0) < spec_repl:
            return False, "statefulset not ready"

    # daemonsets: desiredNumberScheduled == numberAvailable
    for d in ds.get("items",[]):
        status = d.get("status") or {}
        if status.get("desiredNumberScheduled",0) != status.get("numberAvailable",0):
            return False, "daemonset not available"

    return True, "ok"

# ---- stats ----
def compute_stats(samples):
    n = len(samples)
    mean = statistics.fmean(samples) if n else float("nan")
    sd = statistics.pstdev(samples) if n <= 1 else statistics.stdev(samples)
    if n > 1:
        se = sd / math.sqrt(n)
        ci = (mean - 1.96*se, mean + 1.96*se)
    else:
        ci = (float("nan"), float("nan"))
    return mean, sd, ci, n

# ------------- main loop -------------
def run():
    print(f"Benchmarking {APP_NAMESPACE}/{APP_NAME} for {ITERATIONS} iterations")
    with open(CSV_PATH,"w",newline="") as f:
        csv.writer(f).writerow([
            "iteration","target_revision",
            "start_wall_epoch","start_wall_utc","start_wall_local",
            "end_wall_epoch","end_wall_utc","end_wall_local",
            "deploy_started_epoch","deploy_started_utc","deploy_started_local",
            "deploy_finished_epoch","deploy_finished_utc","deploy_finished_local",
            # new stage splits
            "repo_sync_s","manifest_apply_s","resource_creation_s",
            # existing rollup metrics
            "deployment_total_s","rollout_total_s","pod_rollout_s",
            "status","note"
        ])

    results = []

    for i in range(1, ITERATIONS+1):
        rev = TARGET_REVISIONS[i % len(TARGET_REVISIONS)]
        status, note = "ok", ""

        repo_sync_s = manifest_apply_s = resource_creation_s = None
        deploy_total = rollout_total = pod_rollout = None
        deploy_started_ts = deploy_finished_ts = None
        t0 = t_end = None

        try:
            app0 = argocd_app_json()
            if get_target_revision(app0) != rev:
                print(f"[{i}] Set revision -> {rev}")
                kubectl_patch_revision(rev)
            else:
                print(f"[{i}] Already at {rev} (forcing sync).")

            # Start rollout timing
            t0 = time.time()
            argocd_sync()

            # 1) mark ArgoCD operation start/finish (accurate)
            app1 = argocd_app_json()
            deploy_started_ts, deploy_finished_ts = get_op_times(app1)
            if deploy_started_ts and deploy_finished_ts and deploy_finished_ts >= deploy_started_ts:
                deploy_total = deploy_finished_ts - deploy_started_ts

            # 2) Stage splits (approx) by polling app resources list
            #   a) Wait until resources list appears (first item) -> end of repo+render
            ok, resources, t_first_res = wait_until(
                lambda: (len(list_sync_resources(argocd_app_json())) > 0,
                        list_sync_resources(argocd_app_json())),
                timeout_s=600
            )
            if ok:
                repo_sync_s = max(0.0, t_first_res - (deploy_started_ts or t0))
            else:
                note += "no_resources_seen; "

            #   b) Wait until all resources show status==Synced -> end of apply
            def all_synced_fn():
                r = list_sync_resources(argocd_app_json())
                return (all_synced(r), r)
            ok, resources2, t_all_synced = wait_until(all_synced_fn, timeout_s=600)
            if ok and repo_sync_s is not None:
                manifest_apply_s = max(0.0, t_all_synced - t_first_res)

            #   c) Wait until all workloads are ready -> resource creation / pod rollout
            ok, _, t_all_ready = wait_until(workloads_ready, timeout_s=1800, interval_s=1.0)
            if ok and manifest_apply_s is not None:
                resource_creation_s = max(0.0, t_all_ready - t_all_synced)

            # 3) Wait Healthy (rollout total)
            argocd_wait_healthy()
            t_end = time.time()
            rollout_total = max(0.0, t_end - t0)

            # 4) Derived
            if deploy_total is None and (deploy_started_ts and deploy_finished_ts):
                deploy_total = max(0.0, deploy_finished_ts - deploy_started_ts)
            pod_rollout = None
            if rollout_total is not None and deploy_total is not None:
                pod_rollout = max(0.0, rollout_total - deploy_total)

        except subprocess.CalledProcessError as e:
            status = f"error({e.returncode})"
            note = (note + e.stderr.strip())[:300]
            print(f"[{i}] ERROR: {e}\n{e.stderr}")
        except Exception as e:
            status = f"error({type(e).__name__})"
            note = (note + str(e))[:300]
            print(f"[{i}] ERROR: {e}")

        with open(CSV_PATH,"a",newline="") as f:
            csv.writer(f).writerow([
                i, rev,
                fmt2(t0),  iso_utc(t0),  iso_local(t0),
                fmt2(t_end), iso_utc(t_end), iso_local(t_end),
                fmt2(deploy_started_ts), iso_utc(deploy_started_ts), iso_local(deploy_started_ts),
                fmt2(deploy_finished_ts), iso_utc(deploy_finished_ts), iso_local(deploy_finished_ts),
                fmt2(repo_sync_s), fmt2(manifest_apply_s), fmt2(resource_creation_s),
                fmt2(deploy_total), fmt2(rollout_total), fmt2(pod_rollout),
                status, note
            ])

        results.append({
            "repo_sync": repo_sync_s,
            "manifest_apply": manifest_apply_s,
            "resource_creation": resource_creation_s,
            "deploy_total": deploy_total,
            "rollout_total": rollout_total,
            "pod_rollout": pod_rollout,
            "status": status
        })

        time.sleep(1)

    ok_rows = [r for r in results if r["status"] == "ok"]

    def line(name, key):
        vals = [r[key] for r in ok_rows if r[key] is not None]
        if not vals: 
            print(f"{name:<20} n=0")
            return
        mean, sd, (lo,hi), n = compute_stats(vals)
        print(f"{name:<20} {mean:6.2f} ± {sd:4.2f}   [{lo:6.2f}, {hi:6.2f}]   n={n}")

    print("\nSummary (seconds):")
    line("Repo sync", "repo_sync")
    line("Manifest apply", "manifest_apply")
    line("Resource creation", "resource_creation")
    line("Deployment total", "deploy_total")
    line("Pod rollout", "pod_rollout")
    line("Rollout total", "rollout_total")
    print(f"\nSaved CSV -> {CSV_PATH}")

if __name__ == "__main__":
    run()
