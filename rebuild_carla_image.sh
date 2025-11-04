#!/bin/bash
set -e

echo "========================================"
echo "Rebuilding Carla Docker Image"
echo "========================================"
echo ""

# Change this to your Docker Hub username or registry
REGISTRY="fabi200123"
IMAGE_NAME="carla-scheduler"
NEW_TAG="latest"

echo "Building image: $REGISTRY/$IMAGE_NAME:$NEW_TAG"
echo ""

# Build the image
docker build -t $REGISTRY/$IMAGE_NAME:$NEW_TAG -t $REGISTRY/$IMAGE_NAME:latest .

echo ""
echo "✓ Image built successfully"
echo ""

# Ask if they want to push
read -p "Do you want to push to Docker Hub? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Pushing image..."
    docker push $REGISTRY/$IMAGE_NAME:$NEW_TAG
    docker push $REGISTRY/$IMAGE_NAME:latest
    echo "✓ Image pushed"
    
    echo ""
    echo "Now update the Helm values to use the new image:"
    echo "  Image: $REGISTRY/$IMAGE_NAME:$NEW_TAG"
    echo ""
    echo "Then sync ArgoCD:"
    echo "  argocd app sync carla-scheduler"
else
    echo ""
    echo "Image built locally but not pushed."
    echo ""
    echo "For local testing, you can:"
    echo "1. Load the image into your cluster:"
    echo "   docker save $REGISTRY/$IMAGE_NAME:$NEW_TAG | kubectl apply -f -"
    echo ""
    echo "2. Or if using kind/k3s:"
    echo "   kind load docker-image $REGISTRY/$IMAGE_NAME:$NEW_TAG"
    echo "   # or"
    echo "   k3s ctr images import $REGISTRY/$IMAGE_NAME:$NEW_TAG"
fi

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. Update helm/carla-scheduler/values.yaml:"
echo "   Change tag from 'latest' to '$NEW_TAG'"
echo ""
echo "2. Commit and push changes:"
echo "   git add helm/carla-scheduler/values.yaml"
echo "   git commit -m 'Update Carla image to $NEW_TAG'"
echo "   git push origin feat/carla"
echo ""
echo "3. Sync ArgoCD:"
echo "   argocd app sync carla-scheduler"
echo ""


