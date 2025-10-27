# Reproduction Protocol – CARLA Scheduler Deployment

This document describes how the CARLA orchestration component was built, published, and deployed for validation
on an independent K3s/Argo CD environment.  
The procedure applies to any on-prem or cloud Kubernetes cluster running Argo CD ≥ v2.10.

---

## 1. Overview

CARLA (**Clinic-Aware, Risk-Weighted, Load-Aware Orchestrator**) is a lightweight controller that interfaces with
Argo CD to schedule biomedical workloads according to clinical activity, risk score, and cluster load.

The component is distributed publicly via **Docker Hub** to ensure reproducibility without requiring access to
internal package registries.

**Image:** [`alexbozdog/carla-scheduler:latest`](https://hub.docker.com/r/alexbozdog/carla-scheduler)  
**Source:** [fabi200123/biomedical-eac-devsecops](https://github.com/fabi200123/biomedical-eac-devsecops)

---

## 2. Build and Publication

These steps were executed locally  without cluster access.

#### 1. Authenticate to Docker Hub
#### 2. Build the container image from project root
#### 3. Push to Docker Hub for public access

The repository is public, enabling collaborators and reviewers to pull the verified image directly.

---

## 3. Deployment (performed by cluster administrator)
The cluster administrator  deploys CARLA using Argo CD.
The administrator requires:
Access to the target K3s/Kubernetes cluster.
An Argo CD API bearer token (created through the Argo CD UI).

### 3.1. Create namespace and Argo CD token secret
```
kubectl create namespace carla

kubectl -n carla create secret generic carla-argocd-token \
  --from-literal=token='<ARGO_BEARER_TOKEN>'
```
The token is cluster-specific and must never be stored in version control.
It enables CARLA to trigger reconciliations through the Argo CD API.

### 3.2. Deploy via Argo CD Application
```
kubectl apply -f argo-apps/applications/carla/carla-scheduler.yaml
```

---

## 4. Verification

After deployment, verify that CARLA is running:

```
kubectl -n carla get pods
kubectl -n carla logs deploy/carla-scheduler | head
```
Expected output:
[INFO] CARLA scheduler started (tick=30s, Rmax=6.00)

CARLA will then execute its scheduling loop, reading configuration files from
/app/configs/ and writing telemetry to /data/.

---

## 5. Security and Reproducibility Notes

Only the cluster administrator creates and manages secrets (no credentials are stored in the public repo).
The container image is publicly reproducible from the Dockerfile and available on Docker Hub.
All Kubernetes and Helm manifests are version-controlled in the main repository.


Maintainers:
@bozdogalex
@fabi200123


Pull your branch:

git fetch origin feat/carla && git checkout feat/carla


Create the token secret:

kubectl create namespace carla || true
kubectl -n carla create secret generic carla-argocd-token \
  --from-literal=token='<ARGO_BEARER_TOKEN>' --dry-run=client -o yaml | kubectl apply -f -


verify baseUrl in 

helm/values/clusters/in-cluster/Resources/carla-scheduler.yaml


Apply the CARLA Application:

kubectl apply -f argo-apps/applications/carla/carla_scheduler.yaml


Watch it come up:

kubectl -n carla get pods
kubectl -n carla logs deploy/carla-scheduler | head -n 50


You should see: CARLA scheduler started (tick=30s, Rmax=6.00)