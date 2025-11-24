# CARLA Deployment Summary

## ✅ Successfully Synced to ArgoCD

All changes from the working deployment have been committed to the `feat/carla` branch and synced via ArgoCD.

### Changes Applied

#### 1. **Image Configuration**
- Repository: `fabian2001/carla-scheduler`
- Tag: `v1.5`
- Pull Policy: `Always`

#### 2. **Environment Variables**
- `CARLA_POLICY`: `/app/configs/carla_policy.yaml`
- `CARLA_APPS`: `/app/configs/apps.yaml`
- `ARGO_TOKEN`: From secret `carla-argocd-token`
- `PYTHONWARNINGS`: `ignore:Unverified HTTPS request` (suppresses urllib3 warnings)

#### 3. **Argo CD Configuration**
- Base URL: `https://argo-cd-argocd-server.argocd.svc.cluster.local`
- Verify TLS: `false`
- App Namespace: `argocd` (corrected from `mgmt`)

#### 4. **Code Fixes (in ConfigMaps)**
- **Midnight-crossing windows**: Properly handles time windows that cross midnight (20:00-06:00)
- **Sync API**: Added `trigger_sync()` method for proper Argo CD sync operations
- **URL building**: Fixed `_build_app_url()` to correctly append endpoints
- **Toggle revisions**: Disabled to avoid invalid revision errors

### Git Repository Structure

```
biomedical-eac-devsecops/
├── helm/
│   ├── carla-scheduler/
│   │   ├── configs/
│   │   │   ├── apps.yaml              # App registry with toggleRevisions disabled
│   │   │   └── carla_policy.yaml      # Policy with correct Argo CD settings
│   │   └── values.yaml                # Main Helm values
│   └── values/
│       └── clusters/
│           └── in-cluster/
│               └── Resources/
│                   └── carla-scheduler.yaml  # ✅ UPDATED - ArgoCD uses this
└── scripts/
    └── carla_scheduler.py             # Main CARLA scheduler code
```

### Commits on feat/carla Branch

```
8dd969f - chore: sync carla-scheduler values with deployment changes
64307cd - fix: disable toggleRevisions and add proper sync API support
cbc7524 - chore: update carla-scheduler image tag to v1.5 and set imagePullPolicy to Always
958a584 - fix: handle windows that cross midnight and suppress urllib3 warnings
52a99a1 - feat: suppress urllib3 warnings and change sync window to 20:00-06:00
```

### ArgoCD Application

**Name**: `carla-scheduler`
**Namespace**: `argocd`
**Target Revision**: `feat/carla`
**Sync Policy**: Automated
**Status**: ✅ Synced

### Current Deployment Status

```bash
# Check deployment
kubectl get deployment carla-scheduler -n carla

# Check pod
kubectl get pods -n carla -l app.kubernetes.io/name=carla-scheduler

# Check ArgoCD status
kubectl get application carla-scheduler -n argocd

# View logs
kubectl logs -n carla deployment/carla-scheduler -f
```

### CARLA Functionality

✅ **Window Detection**: Correctly identifies 20:00-06:00 (Bucharest time) windows on weekdays
✅ **Sync Operations**: Successfully triggers syncs for out-of-sync applications
✅ **Clean Logs**: No urllib3 InsecureRequestWarning messages
✅ **Argo CD Integration**: Properly communicates with Argo CD API

### Related Files

- **Toggle Script**: `/home/fabian/toggle_metallb_version.sh`
- **Cron Setup**: `/home/fabian/setup_carla_toggle_cron.sh`
- **Documentation**: `/home/fabian/CARLA_TOGGLE_README.md`

## Next Steps

1. Monitor CARLA logs to ensure stable operation
2. Set up the hourly metallb toggle (optional):
   ```bash
   ./setup_carla_toggle_cron.sh
   ```
3. Consider merging `feat/carla` to `main` once stable

---
*Generated: $(date)*
