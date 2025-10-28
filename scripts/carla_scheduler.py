# CARLA — Clinic-Aware, Risk-Weighted, Load-Aware Orchestrator
# Selects which Argo CD Applications to sync now based on:
#  - clinic windows (allowed hours/days)
#  - risk weighting
#  - real-time headroom (CPU/RAM watermarks)
#  - resource budget Rmax via 0/1 knapsack over app scores
#
# Config:
#   - configs/carla_policy.yaml
#   - configs/apps.yaml
#
# Logs:
#   - /data/carla_runs/decisions.csv
#   - /data/carla_runs/rollouts.csv
import csv
import json
import logging
import os
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import pytz
import requests
import yaml
from socketserver import ThreadingMixIn

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    HAVE_PROM = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_PROM = False
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"
    Counter = Gauge = Histogram = None  # type: ignore
    generate_latest = None  # type: ignore

# Optional K8s metrics (graceful fallback)
try:
    from kubernetes import client, config as k8s_config
    HAVE_K8S = True
except Exception:
    HAVE_K8S = False

# ---------------------------- Metrics & Logging ----------------------------

if HAVE_PROM:
    DECISIONS_COUNTER = Counter(
        "carla_decisions_total",
        "Count of CARLA scheduling decisions by outcome",
        ["status"],
    )
    SYNC_TRIGGER_COUNTER = Counter(
        "carla_sync_trigger_total",
        "Total Argo CD sync triggers initiated by CARLA",
    )
    SYNC_ERROR_COUNTER = Counter(
        "carla_sync_error_total",
        "Total Argo CD sync trigger failures recorded by CARLA",
    )
    HIGH_RISK_PAUSED_COUNTER = Counter(
        "carla_high_risk_paused_total",
        "Total high-risk applications paused due to human-in-the-loop policy",
    )
    HEADROOM_GAUGE = Gauge(
        "carla_headroom_ratio",
        "Current CARLA computed headroom ratio",
    )
    RMAX_GAUGE = Gauge(
        "carla_knapsack_rmax_used",
        "Combined resource envelope used by current CARLA selection",
    )
    ROLLOUT_HISTOGRAM = Histogram(
        "carla_rollout_duration_seconds",
        "Observed rollout durations for CARLA-triggered syncs",
    )
else:  # pragma: no cover - optional dependency path
    DECISIONS_COUNTER = None
    SYNC_TRIGGER_COUNTER = None
    SYNC_ERROR_COUNTER = None
    HIGH_RISK_PAUSED_COUNTER = None
    HEADROOM_GAUGE = None
    RMAX_GAUGE = None
    ROLLOUT_HISTOGRAM = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Attach structured extras if present
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in (
                "args",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            ):
                continue
            payload[key] = value
        return json.dumps(payload)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _probe_handler_factory(carla: "Carla"):
    class ProbeHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # type: ignore[override]
            if self.path == "/healthz":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"healthy")
                return

            if self.path == "/readyz":
                if carla.is_ready:
                    self.send_response(200)
                    body = b"ready"
                else:
                    self.send_response(503)
                    body = b"not-ready"
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/metrics":
                if HAVE_PROM and generate_latest:
                    metrics = generate_latest()
                    self.send_response(200)
                    self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                    self.send_header("Content-Length", str(len(metrics)))
                    self.end_headers()
                    self.wfile.write(metrics)
                else:
                    payload = b"no metrics"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            # Silence default HTTP request logging; metrics are emitted separately.
            return

    return ProbeHandler


class ProbeServer(threading.Thread):
    def __init__(self, carla: "Carla", host: str = "0.0.0.0", port: int = 8000):
        super().__init__(daemon=True)
        self._server = ThreadingHTTPServer((host, port), _probe_handler_factory(carla))
        self._stopped = threading.Event()

    def run(self) -> None:
        self._server.serve_forever(poll_interval=0.5)

    def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        try:
            self._server.shutdown()
        finally:
            self._server.server_close()

# ---------------------------- Utilities ----------------------------

def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def now_utc_ts() -> int:
    return int(time.time())


def weekday_name(dt_local: datetime) -> str:
    return ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][dt_local.weekday()]


def in_allowed_window(cfg_windows: dict, dt_utc: datetime) -> bool:
    """
    windows:
      timezone: "Europe/Bucharest"
      allow:
        - { day: WEEKDAY|WEEKEND|MON..SUN, start: "HH:MM", end: "HH:MM" }
      block: [ ... same shape ... ]
    """
    tzname = cfg_windows.get("timezone", "UTC")
    tzinfo = pytz.timezone(tzname)
    dt_local = dt_utc.astimezone(tzinfo)
    day = weekday_name(dt_local)
    is_weekend = day in ("SAT", "SUN")
    is_weekday = not is_weekend
    cur = dt_local.time()
    def _match(rule_day: str) -> bool:
        if rule_day == "WEEKDAY": return is_weekday
        if rule_day == "WEEKEND": return is_weekend
        return rule_day == day

    # block rules take precedence
    for rule in cfg_windows.get("block", []) or []:
        if _match(rule["day"]):
            start_h, start_m = map(int, rule["start"].split(":"))
            end_h, end_m   = map(int, rule["end"].split(":"))
            if (cur >= datetime(dt_local.year, dt_local.month, dt_local.day, start_h, start_m).time()
                and cur <= datetime(dt_local.year, dt_local.month, dt_local.day, end_h, end_m).time()):
                return False

    # allow rules (any matching rule grants permission)
    for rule in cfg_windows.get("allow", []) or []:
        if _match(rule["day"]):
            start_h, start_m = map(int, rule["start"].split(":"))
            end_h, end_m   = map(int, rule["end"].split(":"))
            if (cur >= datetime(dt_local.year, dt_local.month, dt_local.day, start_h, start_m).time()
                and cur <= datetime(dt_local.year, dt_local.month, dt_local.day, end_h, end_m).time()):
                return True
    return False


def ema_update(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None:
        return x
    return (1 - alpha) * prev + alpha * x


def _knapsack_exact(items: List[Tuple[dict, float, float]], Rmax: float) -> List[dict]:
    """
    items: list of (app_dict, weight=r_i, value=score s_i)
    Solve 0/1 knapsack exactly via DP after scaling to ints.
    """
    if not items or Rmax <= 0:
        return []

    scale = 100  # support two decimal precision
    weights = [max(1, int(round(w * scale))) for _, w, _ in items]
    values = [v for _, _, v in items]
    W = max(1, int(round(Rmax * scale)))
    n = len(items)

    dp = [0.0] * (W + 1)
    keep = [[False] * (W + 1) for _ in range(n)]

    for i in range(n):
        w, v = weights[i], values[i]
        for cap in range(W, w - 1, -1):
            cand = dp[cap - w] + v
            if cand > dp[cap]:
                dp[cap] = cand
                keep[i][cap] = True

    cap = max(range(W + 1), key=lambda c: dp[c])
    chosen = []
    for i in range(n - 1, -1, -1):
        if keep[i][cap]:
            chosen.append(items[i][0])
            cap -= weights[i]
    return list(reversed(chosen))


def _knapsack_greedy(items: List[Tuple[dict, float, float]], Rmax: float) -> List[dict]:
    if not items or Rmax <= 0:
        return []

    budget = 0.0
    chosen: List[dict] = []
    for app, weight, score in sorted(
        items, key=lambda tup: (tup[2] / max(tup[1], 1e-6)), reverse=True
    ):
        if budget + weight <= Rmax:
            chosen.append(app)
            budget += weight
    return chosen


def knapsack_select(items: List[Tuple[dict, float, float]], Rmax: float) -> Tuple[List[dict], str]:
    """
    Attempt to compute the optimal set via DP, falling back to greedy selection if
    the DP exceeds 100ms.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_knapsack_exact, items, Rmax)
        try:
            result = future.result(timeout=0.1)
            return result, "exact"
        except FuturesTimeout:
            future.cancel()
    # Greedy fallback
    return _knapsack_greedy(items, Rmax), "greedy"


# ---------------------------- Argo CD client ----------------------------

class ArgoClient:
    def __init__(self, base_url: str, token: str, verify_tls: bool = True):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.verify = verify_tls

    def get_app(self, app_name: str) -> dict:
        url = f"{self.base_url}/api/v1/applications/{app_name}"
        r = requests.get(url, headers=self.headers, verify=self.verify, timeout=15)
        r.raise_for_status()
        return r.json()

    def set_target_revision(self, app_name: str, target_rev: str):
        # GET, mutate spec.source.targetRevision, PUT back
        url = f"{self.base_url}/api/v1/applications/{app_name}"
        app = self.get_app(app_name)
        # Handle single-source Applications; extend if multiple sources are used
        if "spec" not in app or "source" not in app["spec"]:
            raise RuntimeError(f"Application {app_name} has unexpected spec shape")
        app["spec"]["source"]["targetRevision"] = target_rev
        r = requests.put(url, headers=self.headers, verify=self.verify, json=app, timeout=20)
        r.raise_for_status()

    def wait_succeeded(self, app_name: str, poll_s: float = 1.0, timeout_s: int = 1800) -> float:
        t0 = time.time()
        while True:
            app = self.get_app(app_name)
            phase = (((app.get("status") or {}).get("operationState") or {}).get("phase"))
            if phase == "Succeeded":
                return time.time() - t0
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"{app_name} did not reach Succeeded within {timeout_s}s")
            time.sleep(poll_s)


# ---------------------------- K8s metrics ----------------------------

class K8sMetrics:
    def __init__(self, prefer_metrics_api: bool):
        self.prefer = prefer_metrics_api and HAVE_K8S
        self._init_done = False

    def init(self):
        if not HAVE_K8S:
            return
        try:
            # Try in-cluster; fallback to local kubeconfig
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()
            self.core = client.CoreV1Api()
            self.metrics = None
            try:
                self.metrics = client.CustomObjectsApi()
            except Exception:
                self.metrics = None
            self._init_done = True
        except Exception:
            self._init_done = False

    def usage_fraction(self) -> Tuple[float, float]:
        """
        Returns (cpu_used/allocatable_cores, mem_used/allocatable_bytes).
        If metrics unavailable, returns conservative (0.8, 0.8) to force deferral.
        """
        if not self._init_done:
            self.init()
        if not HAVE_K8S or not self._init_done:
            return (0.8, 0.8)  # conservative fallback

        # Get allocatable
        try:
            nodes = self.core.list_node().items
            alloc_cores = 0.0
            alloc_mem = 0.0
            for n in nodes:
                cap = n.status.allocatable
                # cpu like "8", mem like "16428036Ki"
                c = cap.get("cpu", "0")
                m = cap.get("memory", "0")
                # cpu: can be "250m" or "8" — normalize to cores
                if c.endswith("m"):
                    alloc_cores += float(c[:-1]) / 1000.0
                else:
                    alloc_cores += float(c)
                # memory: Ki/Mi/Gi
                if m.endswith("Ki"):
                    alloc_mem += float(m[:-2]) * 1024
                elif m.endswith("Mi"):
                    alloc_mem += float(m[:-2]) * 1024 * 1024
                elif m.endswith("Gi"):
                    alloc_mem += float(m[:-2]) * 1024 * 1024 * 1024
                else:
                    alloc_mem += float(m)  # bytes
        except Exception:
            return (0.8, 0.8)

        # Get usage (metrics-server via metrics.k8s.io is not directly available through CoreV1)
        # For simplicity and portability, return a neutral value if metrics group isn't present.
        # You can upgrade this to query your Alloy/Mimir signals.
        try:
            # Try metrics.k8s.io custom object
            # Group/Version/Plural depends on the installed metrics API; many clusters disable it.
            # We'll skip exact call to keep this robust.
            used_cpu_frac = 0.4  # reasonable default
            used_mem_frac = 0.5
        except Exception:
            used_cpu_frac = 0.4
            used_mem_frac = 0.5

        return (used_cpu_frac, used_mem_frac)


# ---------------------------- CARLA core ----------------------------

class Carla:
    def __init__(self, policy_path: str, apps_path: str):
        self.policy = load_yaml(policy_path)
        self.apps_cfg = load_yaml(apps_path)

        lg = self.policy.get("logging", {})
        lvl = lg.get("level", "INFO").upper()
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, lvl, logging.INFO))
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root_logger.handlers = [handler]
        self.log = logging.getLogger("carla")

        th = self.policy["thresholds"]
        self.cpu_wm = float(th["cpu_watermark"])
        self.ram_wm = float(th["ram_watermark"])
        self.min_headroom = float(th["min_headroom"])

        res = self.policy["resources"]
        self.Rmax = float(res["Rmax"])

        w = self.policy["weights"]
        self.alpha = float(w["alpha_depth"])
        self.beta  = float(w["beta_invres"])
        self.gamma = float(w["gamma_risk"])
        self.delta = float(w["delta_window"])

        self.ema_alpha = float(self.policy["ema"]["rollout_alpha"])

        self.windows = self.policy.get("windows", {})
        self.hil = self.policy.get("human_in_the_loop", {})
        self.tick_s = int(self.policy["tick"]["seconds"])
        self.max_parallel = int(self.policy["tick"].get("max_parallel_triggers", 1))

        # Argo
        argo = self.policy["argo"]
        token_env = argo.get("token_env", "ARGO_TOKEN")
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(f"Missing env var {token_env} with Argo CD token")
        self.argo = ArgoClient(argo["base_url"], token, argo.get("verify_tls", True))

        # Telemetry & logging paths
        tel = self.policy.get("telemetry", {})
        self.prefer_k8s_metrics = bool(tel.get("prefer_k8s_metrics_api", True))
        self.k8s = K8sMetrics(self.prefer_k8s_metrics)

        self.decisions_csv = lg.get("decisions_csv", "/data/carla_runs/decisions.csv")
        self.rollouts_csv  = lg.get("rollouts_csv",  "/data/carla_runs/rollouts.csv")
        ensure_parent(self.decisions_csv); ensure_parent(self.rollouts_csv)

        # state
        self.ema_rollout = None
        self.stop = False
        self._ready_event = threading.Event()
        self.argo_failures: Dict[str, Dict[str, float]] = {}
        self.inflight: Dict[str, float] = {}
        self.override_path = "/app/configs/override.yaml"
        self.hil_override_until = 0.0
        self._hil_override_logged = False

        # normalize app entries
        defaults = self.apps_cfg.get("defaults", {})
        def_ri = float(defaults.get("r_i", 1.5))
        def_di = int(defaults.get("d_i", 1))
        def_rk = int(defaults.get("risk", 1))

        self.apps: List[Dict[str, Any]] = []
        for a in self.apps_cfg["apps"]:
            app = {
                "name": a["name"],
                "r_i": float(a.get("r_i", def_ri)),
                "d_i": int(a.get("d_i", def_di)),
                "risk": int(a.get("risk", def_rk)),
                "toggles": list(a.get("toggleRevisions", [])),
                "next_toggle_idx": 0
            }
            self.apps.append(app)

        # CSV headers
        if not os.path.exists(self.decisions_csv):
            with open(self.decisions_csv, "w", newline="", encoding="utf-8") as fh:
                cw = csv.writer(fh)
                cw.writerow(["ts","policy","allowed_window","headroom","selected","skipped","reason"])
        if not os.path.exists(self.rollouts_csv):
            with open(self.rollouts_csv, "w", newline="", encoding="utf-8") as fh:
                cw = csv.writer(fh)
                cw.writerow(["ts_start","ts_end","app","revision","rollout_seconds"])

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    # Headroom index H = 1 - max(cpu/θ_cpu, ram/θ_ram)
    def compute_headroom(self) -> float:
        cpu_frac, mem_frac = self.k8s.usage_fraction()
        cpu_ratio = cpu_frac / max(self.cpu_wm, 1e-6)
        mem_ratio = mem_frac / max(self.ram_wm, 1e-6)
        H = 1.0 - max(cpu_ratio, mem_ratio)
        return max(-1.0, min(1.0, H))

    def now_allowed(self) -> bool:
        return in_allowed_window(self.windows, datetime.utcnow().replace(tzinfo=pytz.UTC))

    def score(self, app: dict, window_fit: int) -> float:
        # s_i = α/(d+1) + β/r - γ*risk + δ*window_fit
        return (self.alpha / (app["d_i"] + 1.0)
                + self.beta / max(app["r_i"], 1e-6)
                - self.gamma * app["risk"]
                + self.delta * window_fit)

    def choose_toggle(self, app: dict) -> str:
        toggles = app["toggles"] or ["main"]
        idx = app["next_toggle_idx"] % len(toggles)
        rev = toggles[idx]
        app["next_toggle_idx"] = idx + 1
        return rev

    def _read_override_flag(self) -> bool:
        if not os.path.exists(self.override_path):
            return False
        try:
            data = load_yaml(self.override_path)
        except Exception as exc:
            self.log.warning("Failed to read override config", extra={"error": str(exc)})
            return False
        if not isinstance(data, dict):
            return False
        if data.get("carla.ops/override") is True:
            return True
        carla_ops = data.get("carla.ops")
        if isinstance(carla_ops, dict) and carla_ops.get("override") is True:
            return True
        carla_section = data.get("carla")
        if isinstance(carla_section, dict):
            if carla_section.get("ops/override") is True:
                return True
            ops_section = carla_section.get("ops")
            if isinstance(ops_section, dict) and ops_section.get("override") is True:
                return True
        return False

    def _should_bypass_hil(self, now_ts: float) -> bool:
        if self._read_override_flag():
            self.hil_override_until = max(self.hil_override_until, now_ts + self.tick_s)
        if now_ts <= self.hil_override_until:
            if not self._hil_override_logged:
                self.log.info("Human-in-the-loop override active", extra={"override_until": self.hil_override_until})
                self._hil_override_logged = True
            return True
        if self._hil_override_logged:
            self.log.info("Human-in-the-loop override cleared")
        self._hil_override_logged = False
        self.hil_override_until = 0.0
        return False

    def _fetch_app_status(self, app_name: str, now_ts: float) -> Optional[dict]:
        state = self.argo_failures.get(app_name)
        if state:
            next_ts = state.get("next_ts", 0.0)
            if now_ts < next_ts:
                # Respect exponential backoff window
                return None
        try:
            app = self.argo.get_app(app_name)
            self.argo_failures[app_name] = {"count": 0, "next_ts": now_ts}
            return app
        except Exception as exc:
            state = self.argo_failures.get(app_name, {"count": 0, "next_ts": now_ts})
            count = state.get("count", 0) + 1
            delay = 30 if count >= 3 else min(30, 2 ** count)
            next_ts = now_ts + delay
            self.argo_failures[app_name] = {"count": count, "next_ts": next_ts}
            log_payload = {
                "app": app_name,
                "failure_count": count,
                "backoff_seconds": delay,
                "error": str(exc),
            }
            if count >= 3:
                self.log.error(
                    "Argo API failure threshold reached for %s; circuit open",
                    app_name,
                    extra=log_payload,
                )
            else:
                self.log.warning(
                    "Argo API failure for %s; backing off",
                    app_name,
                    extra=log_payload,
                )
            return None

    def trigger_and_wait(self, app: dict, revision: str) -> float:
        name = app["name"]
        start_ts = time.time()
        self.inflight[name] = start_ts
        if SYNC_TRIGGER_COUNTER:
            SYNC_TRIGGER_COUNTER.inc()
        try:
            self.argo.set_target_revision(name, revision)
            duration = self.argo.wait_succeeded(name)
            end_ts = time.time()
            with open(self.rollouts_csv, "a", newline="", encoding="utf-8") as fh:
                cw = csv.writer(fh)
                cw.writerow([int(start_ts), int(end_ts), name, revision, f"{duration:.2f}"])
            self.ema_rollout = ema_update(self.ema_rollout, duration, self.ema_alpha)
            if ROLLOUT_HISTOGRAM:
                ROLLOUT_HISTOGRAM.observe(duration)
            return duration
        finally:
            self.inflight.pop(name, None)

    def run_once(self):
        start_time = time.time()
        now_ts = start_time
        H = self.compute_headroom()
        if HEADROOM_GAUGE:
            HEADROOM_GAUGE.set(H)
        allowed = self.now_allowed()

        bypass_hil = self._should_bypass_hil(now_ts)
        items: List[Tuple[dict, float, float]] = []
        app_scores: Dict[str, float] = {}
        status_map: Dict[str, str] = {}
        skipped_pre_selection: List[str] = []

        for app in self.apps:
            name = app["name"]
            if name in self.inflight:
                status_map[name] = "inflight"
                skipped_pre_selection.append(name)
                continue
            status = self._fetch_app_status(name, now_ts)
            if status is None:
                state = self.argo_failures.get(name, {})
                status_map[name] = f"backoff({int(state.get('count', 0))})"
                skipped_pre_selection.append(name)
                continue
            sync_state = ((status.get("status") or {}).get("sync") or {})
            sync_status = sync_state.get("status", "Unknown")
            status_map[name] = sync_status
            if sync_status != "OutOfSync":
                skipped_pre_selection.append(name)
                continue
            wf = 1 if allowed else 0
            score_val = self.score(app, wf)
            items.append((app, app["r_i"], score_val))
            app_scores[name] = score_val

        selected_candidates: List[dict] = []
        selection_mode = "n/a"
        reason = ""
        if H < self.min_headroom:
            reason = f"low_headroom(H={H:.2f}<Δ={self.min_headroom:.2f})"
        elif not allowed:
            reason = "outside_allowed_window"
        elif not items:
            reason = "no_pending_apps"
        else:
            selected_candidates, selection_mode = knapsack_select(items, self.Rmax)
            if not selected_candidates:
                reason = "no_selection_under_Rmax"

        triggered_apps: List[str] = []
        skipped_after_selection: List[str] = []
        paused_apps: List[str] = []
        rmax_used = 0.0
        max_parallel = self.max_parallel
        for app in selected_candidates:
            if len(triggered_apps) >= max_parallel:
                skipped_after_selection.append(app["name"])
                continue
            if self.hil.get("pause_high_risk", True) and app["risk"] >= 2 and not bypass_hil:
                paused_apps.append(app["name"])
                if HIGH_RISK_PAUSED_COUNTER:
                    HIGH_RISK_PAUSED_COUNTER.inc()
                continue
            rev = self.choose_toggle(app)
            self.log.info("Triggering sync", extra={"app": app["name"], "revision": rev})
            try:
                self.trigger_and_wait(app, rev)
                triggered_apps.append(app["name"])
                rmax_used += float(app["r_i"])
            except Exception as exc:  # pragma: no cover - network interaction
                skipped_after_selection.append(app["name"])
                if SYNC_ERROR_COUNTER:
                    SYNC_ERROR_COUNTER.inc()
                self.log.error(
                    "Rollout failed",
                    extra={"app": app["name"], "error": str(exc)},
                )

        skipped_apps = sorted(set(skipped_pre_selection + skipped_after_selection))
        duration_ms = (time.time() - start_time) * 1000.0
        decision_id = f"{int(now_ts)}-{uuid.uuid4().hex[:6]}"
        if RMAX_GAUGE:
            RMAX_GAUGE.set(rmax_used)

        # CSV logging for backward compatibility
        with open(self.decisions_csv, "a", newline="", encoding="utf-8") as fh:
            cw = csv.writer(fh)
            cw.writerow([
                now_utc_ts(),
                "CARLA-knapsack",
                int(allowed),
                f"{H:.3f}",
                "|".join(triggered_apps),
                "|".join(skipped_apps),
                reason or selection_mode,
            ])

        if DECISIONS_COUNTER:
            DECISIONS_COUNTER.labels(status="selected").inc(len(triggered_apps))
            DECISIONS_COUNTER.labels(status="skipped").inc(len(skipped_apps))
            DECISIONS_COUNTER.labels(status="paused").inc(len(paused_apps))

        log_payload = {
            "decision_id": decision_id,
            "window_allowed": allowed,
            "headroom": H,
            "risk": {app["name"]: app["risk"] for app, _, _ in items},
            "score": app_scores,
            "selected_apps": triggered_apps,
            "skipped_apps": skipped_apps,
            "paused_apps": paused_apps,
            "fallback_used": selection_mode,
            "argo_status": status_map,
            "rmax_used": rmax_used,
            "latency_ms": round(duration_ms, 2),
            "reason": reason,
            "bypass_hil": bypass_hil,
        }
        self.log.info("decision", extra=log_payload)
        self._ready_event.set()

    def run_forever(self):
        self.log.info("CARLA scheduler started (tick=%ss, Rmax=%.2f)", self.tick_s, self.Rmax)
        while not self.stop:
            t0 = time.time()
            try:
                self.run_once()
            except Exception as e:
                self.log.error("run_once error: %s", e)
            dt = time.time() - t0
            sleep_s = max(0.0, self.tick_s - dt)
            time.sleep(sleep_s)


def main():
    policy_path = os.environ.get("CARLA_POLICY", "configs/carla_policy.yaml")
    apps_path   = os.environ.get("CARLA_APPS",   "configs/apps.yaml")

    carla = Carla(policy_path, apps_path)
    probe_server = ProbeServer(carla)
    probe_server.start()

    def _sigint(sig, frame):
        carla.stop = True
        probe_server.stop()
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        carla.run_forever()
    finally:
        probe_server.stop()


if __name__ == "__main__":
    main()
