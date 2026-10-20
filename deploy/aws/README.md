# Enterprise Knowledge Assistant — AWS Deployment

Production deployment on AWS using ECS Fargate, S3, and CloudFront.

## Architecture

```
                ┌─────────────────┐
                │   CloudFront    │  (HTTPS, CDN)
                │   Distribution  │
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │  S3 Static Site │  (Phase 1 frontend: HTML/CSS/JS)
                │  (eka-frontend) │
                └─────────────────┘

                         │ (REST API calls)
                         ▼
                ┌─────────────────┐
                │  Application    │  (Public, HTTPS)
                │  Load Balancer  │
                └────────┬────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
       ┌──────┐     ┌──────┐    ┌──────┐
       │ ECS  │     │ ECS  │    │ ECS  │  (Auto-scaled 1-10)
       │Task 1│     │Task 2│    │Task N│
       └──────┘     └──────┘    └──────┘
            │            │            │
            └────────────┴────────────┘
                         │
                ┌────────▼────────┐
                │   OpenAI API    │  (or local TemplateLLM if no key)
                │   (external)     │
                └─────────────────┘
```

## Prerequisites

1. **AWS CLI** configured with admin credentials: `aws configure`
2. **Docker** installed and running
3. **AWS region**: default is `us-east-1`. Override with `AWS_REGION=us-west-2`
4. **OpenAI API key** (optional — leave blank to use local TemplateLLM)

## Deploy

### Option A: Full deploy (one command)

```bash
# 1. Create the CloudFormation stack (first time only)
aws cloudformation deploy \
  --template-file deploy/aws/cloudformation.yaml \
  --stack-name eka-prod \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides EnvironmentName=prod OpenAIApiKey=sk-your-key

# 2. Build, push, and deploy
cd deploy/aws && ./deploy.sh prod
```

### Option B: Manual steps

```bash
# 1. Auth to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com

# 2. Build, tag, push
docker build -t eka-backend backend/
docker tag eka-backend:latest <account>.dkr.ecr.us-east-1.amazonaws.com/prod/eka-backend:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/prod/eka-backend:latest

# 3. Sync frontend to S3
aws s3 sync ./ s3://prod-eka-frontend-<account>-us-east-1/ \
  --exclude "*" --include "*.html" --include "css/*" --include "js/*" \
  --exclude "backend/*" --exclude "deploy/*" --exclude ".git/*" --delete

# 4. Force new ECS deployment
aws ecs update-service --cluster prod-eka-cluster --service prod-eka-service --force-new-deployment
```

## What gets created

| Resource | Purpose | Cost (us-east-1, monthly) |
|----------|---------|---------------------------|
| VPC + 2 public subnets | Network isolation | ~$0 |
| Application Load Balancer | HTTPS termination, routing | $16 + $0.008/GB |
| ECS Fargate (2 tasks, 1 vCPU / 2GB each) | Backend containers | $30 |
| ECR repository | Container image storage | $0.10/GB |
| S3 bucket (frontend) | Static HTML/CSS/JS hosting | $0.023/GB |
| CloudFront distribution | CDN + HTTPS for frontend | $0.085/GB |
| CloudWatch logs | Container logs (30-day retention) | $0.50/GB |
| **Estimated total** | (light traffic) | **~$50/month** |

## Auto-scaling

ECS auto-scales based on CPU utilization:
- **Target**: 70% CPU utilization per task
- **Min**: 1 task
- **Max**: 10 tasks
- **Scale-out cooldown**: 60s
- **Scale-in cooldown**: 300s

## Observability

### Logs
```bash
# Tail live logs
aws logs tail /ecs/prod-eka --follow

# Search for errors in last hour
aws logs filter-log-events \
  --log-group-name /ecs/prod-eka \
  --filter-pattern "ERROR" \
  --start-time $(date -d '1 hour ago' +%s)000
```

### Metrics
The backend exposes a `/api/metrics` endpoint in Prometheus format. To scrape it:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'eka-backend'
    dns_sd_configs:
      - names: ['internal-eka-alb-xxx.us-east-1.elb.amazonaws.com']
        type: A
        port: 80
    metrics_path: /api/metrics
```

### LangSmith tracing
If `LANGSMITH_API_KEY` is set, all RAG calls are automatically traced and visible at [smith.langchain.com](https://smith.langchain.com).

## Health checks

- ALB health check: `GET /api/health` (every 30s, 10s timeout, 2 healthy / 3 unhealthy)
- Container health check: same endpoint via Docker `HEALTHCHECK`

## Tear down

```bash
# Delete the CloudFormation stack (cleans up everything except S3 frontend bucket)
aws cloudformation delete-stack --stack-name eka-prod

# Empty + delete the S3 bucket manually (CloudFront cache must be invalidated first)
aws s3 rm s3://prod-eka-frontend-<account>-us-east-1 --recursive
aws s3api delete-bucket --bucket prod-eka-frontend-<account>-us-east-1
```
