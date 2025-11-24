# Repository Guidelines

## Project Structure & Module Organization
The repo is split between Python automation in `scripts/`, declarative manifests in `charts/`, `helm/`, `configs/`, and run artifacts in `reports/`. Generated CARLA data lives under `reports/<tag>` (for example `reports/carla_nov`), while reusable inputs such as policies and app catalogs sit in `configs/`. Kubernetes-facing assets (`argo-apps/`, `pangolin/`, `docs/`) mirror the deployment surface but rarely change when iterating on analytics code. Keep large datasets in `data/` and treat the virtual environment in `venv/` as disposable.

## Build, Test, and Development Commands
Create a sandboxed interpreter per session:
```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```
Dry-run argument parsing or emit metrics with Prometheus + CSV inputs:
```bash
python scripts/carla_metrics_report.py --dry-run
python scripts/carla_metrics_report.py --prom-url http://prom:9090 --start 2024-11-04T20:00:00Z --end 2024-11-05T08:00:00Z --rollouts-csv data/carla_runs/rollouts.csv --outdir reports/carla_nov
```
Offline scheduler fixtures can be replayed with `python scripts/carla_offline_tests.py` to regenerate sample decision and rollout logs.

## Coding Style & Naming Conventions
Python sources target 3.10+, use 4-space indentation, and follow PEP 8 prose (lower_snake_case for functions, UpperCamelCase for classes). Favor explicit typing imports already present in `scripts/carla_metrics_report.py`. Configuration YAML keys stay lowercase with hyphen-free words (see `configs/`). Use docstrings for exported helpers and keep plotting utilities side-effect free except for file writes.

## Testing Guidelines
Prometheus integrations rely on live data, so unit-level validation focuses on offline fixtures. Keep synthetic CSVs in `data/carla_runs/` and ensure they include `rollout_seconds` or similarly named columns for compatibility with the metrics tooling. When adding scheduler behavior, augment `scripts/carla_offline_tests.py` or `scripts/carla_api_dryrun_tests.py` with representative edge cases. Capture baseline artifacts in `reports/<tag>` so reviewers can diff graphs and summaries.

## Commit & Pull Request Guidelines
Stick to short, present-tense commit subjects (`carla_metrics_report script added`, `Apply CARLA review improvements`). Reference related charts or configs in the body when changes span multiple subsystems. Pull requests should mention target environments, attach snippets or figures from `reports/`, and link Prometheus windows or Argo runs that motivated the change. Include reproduction commands plus any required secrets or feature flags so another contributor can regenerate the artifacts locally.
