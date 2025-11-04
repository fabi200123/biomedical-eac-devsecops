#!/bin/bash
set -e

echo "================================"
echo "CARLA ArgoCD Permission Setup"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Create carla-scheduler account in ArgoCD
echo -e "${YELLOW}Step 1: Creating carla-scheduler account in ArgoCD...${NC}"
kubectl patch configmap argocd-cm -n argocd --type merge -p '{
  "data": {
    "accounts.carla-scheduler": "apiKey, login"
  }
}'
echo -e "${GREEN}✓ Account created${NC}"
echo ""

# Step 2: Grant permissions
echo -e "${YELLOW}Step 2: Granting sync permissions to carla-scheduler...${NC}"

# Get existing RBAC policy
EXISTING_POLICY=$(kubectl get configmap argocd-rbac-cm -n argocd -o jsonpath='{.data.policy\.csv}' 2>/dev/null || echo "")

# Add Carla permissions
CARLA_POLICY="p, role:carla-scheduler, applications, get, */*, allow
p, role:carla-scheduler, applications, sync, */*, allow
p, role:carla-scheduler, applications, update, */*, allow
p, role:carla-scheduler, applications, override, */*, allow
g, carla-scheduler, role:carla-scheduler"

# Combine policies
if [ -z "$EXISTING_POLICY" ]; then
  NEW_POLICY="$CARLA_POLICY"
else
  NEW_POLICY="${EXISTING_POLICY}
${CARLA_POLICY}"
fi

kubectl patch configmap argocd-rbac-cm -n argocd --type merge -p "{
  \"data\": {
    \"policy.csv\": \"$NEW_POLICY\"
  }
}"
echo -e "${GREEN}✓ Permissions granted${NC}"
echo ""

# Step 3: Restart ArgoCD server
echo -e "${YELLOW}Step 3: Restarting ArgoCD server to apply changes...${NC}"
kubectl rollout restart deployment argo-cd-argocd-server -n argocd
kubectl rollout status deployment argo-cd-argocd-server -n argocd --timeout=120s
echo -e "${GREEN}✓ ArgoCD server restarted${NC}"
echo ""

# Step 4: Generate token
echo -e "${YELLOW}Step 4: Generating token for carla-scheduler account...${NC}"

# Get admin password
ARGOCD_PASS=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" 2>/dev/null | base64 -d)
if [ -z "$ARGOCD_PASS" ]; then
  echo -e "${RED}✗ Could not retrieve ArgoCD admin password${NC}"
  echo "Please manually generate a token and create the secret:"
  echo "  kubectl -n carla create secret generic carla-argocd-token --from-literal=token='YOUR_TOKEN'"
  exit 1
fi

# Port-forward ArgoCD
echo "Starting port-forward to ArgoCD..."
kubectl port-forward svc/argo-cd-argocd-server -n argocd 8080:443 > /dev/null 2>&1 &
PF_PID=$!
sleep 5

# Login and generate token
echo "Logging in to ArgoCD..."
argocd login localhost:8080 --username admin --password "$ARGOCD_PASS" --insecure

echo "Generating token for carla-scheduler..."
CARLA_TOKEN=$(argocd account generate-token --account carla-scheduler)

# Kill port-forward
kill $PF_PID 2>/dev/null || true

if [ -z "$CARLA_TOKEN" ]; then
  echo -e "${RED}✗ Failed to generate token${NC}"
  exit 1
fi

echo -e "${GREEN}✓ Token generated${NC}"
echo ""

# Step 5: Update secret
echo -e "${YELLOW}Step 5: Updating carla-argocd-token secret...${NC}"
kubectl delete secret carla-argocd-token -n carla 2>/dev/null || true
kubectl -n carla create secret generic carla-argocd-token \
  --from-literal=token="$CARLA_TOKEN"
echo -e "${GREEN}✓ Secret updated${NC}"
echo ""

# Step 6: Restart Carla
echo -e "${YELLOW}Step 6: Restarting Carla scheduler...${NC}"
kubectl rollout restart deployment carla-scheduler -n carla
kubectl rollout status deployment carla-scheduler -n carla --timeout=120s
echo -e "${GREEN}✓ Carla restarted${NC}"
echo ""

# Step 7: Verify
echo -e "${YELLOW}Step 7: Verifying setup...${NC}"
echo "Waiting for Carla to start..."
sleep 10

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Check Carla logs with:"
echo "  kubectl logs -n carla -l app.kubernetes.io/name=carla-scheduler -f"
echo ""
echo "You should see successful sync messages instead of 403 errors."


