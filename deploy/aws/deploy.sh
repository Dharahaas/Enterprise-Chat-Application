#!/usr/bin/env bash
# Enterprise Knowledge Assistant — AWS deployment script
# Builds the backend Docker image, pushes to ECR, syncs frontend to S3,
# and triggers an ECS service redeployment.
#
# Usage:
#   ./deploy/aws/deploy.sh prod
#
# Prerequisites:
#   - aws CLI configured
#   - Docker installed
#   - CloudFormation stack already created (see cloudformation.yaml)

set -euo pipefail

ENV="${1:-prod}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ENV}/eka-backend"

echo "=========================================================="
echo "  Enterprise Knowledge Assistant — AWS Deploy"
echo "  Environment: ${ENV}"
echo "  Region:       ${REGION}"
echo "  ECR URI:      ${ECR_URI}"
echo "=========================================================="

# Step 1: Auth to ECR
echo ""
echo "[1/5] Authenticating to ECR..."
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Step 2: Build the image
echo ""
echo "[2/5] Building Docker image..."
docker build -t eka-backend "${BASH_SOURCE[0]%/*}/../../backend"

# Step 3: Tag + push
echo ""
echo "[3/5] Tagging and pushing to ECR..."
docker tag eka-backend:latest "${ECR_URI}:latest"
docker tag eka-backend:latest "${ECR_URI}:$(date +%Y%m%d-%H%M%S)"
docker push "${ECR_URI}:latest"

# Step 4: Sync frontend to S3
FRONTEND_BUCKET="${ENV}-eka-frontend-${ACCOUNT_ID}-${REGION}"
echo ""
echo "[4/5] Syncing frontend to S3 bucket: ${FRONTEND_BUCKET}"
aws s3 sync "${BASH_SOURCE[0]%/*}/../../" "s3://${FRONTEND_BUCKET}/" \
    --exclude "*" \
    --include "*.html" \
    --include "css/*" \
    --include "js/*" \
    --include "assets/*" \
    --exclude "backend/*" \
    --exclude "deploy/*" \
    --exclude ".git/*" \
    --delete

# Step 5: Force new ECS deployment
CLUSTER_NAME="${ENV}-eka-cluster"
SERVICE_NAME="${ENV}-eka-service"
echo ""
echo "[5/5] Forcing ECS redeployment: ${CLUSTER_NAME}/${SERVICE_NAME}"
aws ecs update-service \
    --cluster "${CLUSTER_NAME}" \
    --service "${SERVICE_NAME}" \
    --force-new-deployment

echo ""
echo "=========================================================="
echo "  ✓ Deployment triggered"
echo ""
echo "  Backend API:  http://$(aws elbv2 describe-load-balancers \
        --names ${ENV}-eka-alb --query 'LoadBalancers[0].DNSName' --output text)"
echo "  Frontend:     https://$(aws cloudfront list-distributions \
        --query 'DistributionList.Items[?starts_with(DomainName,`d`) == `true`].DomainName' \
        --output text | head -1)"
echo ""
echo "  Tail logs:    aws logs tail /ecs/${ENV}-eka --follow"
echo "=========================================================="
