#!/bin/bash

echo "========================================"
echo "Fix Docker Permissions & Rebuild Carla"
echo "========================================"
echo ""

# Check if running on the k8s host
if [ "$(hostname)" = "k8s" ]; then
    echo "✓ Running on k8s host"
else
    echo "⚠ Warning: Not on k8s host. This script expects to run on the cluster node."
fi
echo ""

# Check if docker command exists
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found!"
    exit 1
fi

# Check current user
CURRENT_USER=$(whoami)
echo "Current user: $CURRENT_USER"
echo ""

# Option 1: Add user to docker group (persistent fix)
echo "Option 1: Add user to docker group (recommended)"
echo "This will allow you to run docker without sudo permanently."
echo ""
read -p "Add $CURRENT_USER to docker group? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Adding $CURRENT_USER to docker group..."
    sudo usermod -aG docker $CURRENT_USER
    echo "✓ User added to docker group"
    echo ""
    echo "⚠ You need to log out and back in (or run: newgrp docker)"
    echo "   for the group change to take effect."
    echo ""
    read -p "Run 'newgrp docker' now and continue? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Switching to docker group and rebuilding..."
        exec sg docker "$0 --build-only"
    else
        echo "Please log out and back in, then run this script again."
        exit 0
    fi
fi

# Option 2: Use sudo (one-time)
echo ""
echo "Option 2: Use sudo for this build only"
read -p "Build with sudo? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    USE_SUDO="sudo"
else
    echo "Cancelled."
    exit 0
fi

# Build with sudo
echo ""
echo "========================================"
echo "Building Carla Image"
echo "========================================"
echo ""

REGISTRY="fabi200123"
IMAGE_NAME="carla-scheduler"
TAG="latest"

echo "Building: $REGISTRY/$IMAGE_NAME:$TAG"
echo ""

$USE_SUDO docker build -t $REGISTRY/$IMAGE_NAME:$TAG .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Image built successfully!"
    echo ""
    
    # For k3s, import the image directly
    if command -v k3s &> /dev/null; then
        echo "Detected k3s. Importing image into k3s..."
        $USE_SUDO docker save $REGISTRY/$IMAGE_NAME:$TAG | $USE_SUDO k3s ctr images import -
        if [ $? -eq 0 ]; then
            echo "✓ Image imported into k3s"
        else
            echo "⚠ Failed to import into k3s, but image is built"
        fi
    fi
    
    echo ""
    echo "========================================"
    echo "Next Steps:"
    echo "========================================"
    echo ""
    echo "1. Restart the Carla pod to use the new image:"
    echo "   kubectl delete pod -n carla -l app.kubernetes.io/name=carla-scheduler"
    echo ""
    echo "2. Watch it come back up:"
    echo "   kubectl get pods -n carla -w"
    echo ""
    echo "3. Check the logs:"
    echo "   kubectl logs -n carla -l app.kubernetes.io/name=carla-scheduler -f"
    echo ""
    echo "The new image includes the app_namespace fix!"
    echo ""
else
    echo "❌ Build failed!"
    exit 1
fi


