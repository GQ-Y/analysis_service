# Makefile for Analysis Service

# 变量定义
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := analysis-service
IMAGE_NAME := $(PROJECT_NAME)
VERSION := $(shell grep version pyproject.toml | cut -d'"' -f2)

# 颜色定义
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

.PHONY: help install dev prod test clean docker build deploy health check lint format

# 默认目标
help: ## 显示帮助信息
	@echo "$(BLUE)Analysis Service - 可用命令:$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# 开发环境
install: ## 安装依赖
	@echo "$(YELLOW)安装项目依赖...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)依赖安装完成$(RESET)"

dev: ## 启动开发环境
	@echo "$(YELLOW)启动开发环境...$(RESET)"
	$(PYTHON) run.py

dev-script: ## 使用启动脚本启动开发环境
	@echo "$(YELLOW)使用启动脚本启动开发环境...$(RESET)"
	$(PYTHON) scripts/start.py --mode=dev

prod: ## 启动生产环境
	@echo "$(YELLOW)启动生产环境...$(RESET)"
	$(PYTHON) scripts/start.py --mode=prod

gunicorn: ## 使用Gunicorn启动
	@echo "$(YELLOW)使用Gunicorn启动...$(RESET)"
	$(PYTHON) scripts/start.py --mode=gunicorn

# 测试相关
test: ## 运行测试
	@echo "$(YELLOW)运行测试...$(RESET)"
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

test-unit: ## 运行单元测试
	@echo "$(YELLOW)运行单元测试...$(RESET)"
	pytest tests/unit/ -v

test-integration: ## 运行集成测试
	@echo "$(YELLOW)运行集成测试...$(RESET)"
	pytest tests/integration/ -v

test-coverage: ## 生成测试覆盖率报告
	@echo "$(YELLOW)生成测试覆盖率报告...$(RESET)"
	pytest tests/ --cov=app --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)覆盖率报告已生成到 htmlcov/index.html$(RESET)"

# 代码质量
lint: ## 代码检查
	@echo "$(YELLOW)运行代码检查...$(RESET)"
	flake8 app/ --max-line-length=120
	pylint app/ --max-line-length=120
	mypy app/ --ignore-missing-imports

format: ## 代码格式化
	@echo "$(YELLOW)格式化代码...$(RESET)"
	black app/ --line-length=120
	isort app/ --profile=black

format-check: ## 检查代码格式
	@echo "$(YELLOW)检查代码格式...$(RESET)"
	black app/ --check --line-length=120
	isort app/ --check-only --profile=black

# Docker相关
docker-build: ## 构建Docker镜像
	@echo "$(YELLOW)构建Docker镜像...$(RESET)"
	$(DOCKER) build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest .
	@echo "$(GREEN)Docker镜像构建完成$(RESET)"

docker-build-dev: ## 构建开发环境Docker镜像
	@echo "$(YELLOW)构建开发环境Docker镜像...$(RESET)"
	$(DOCKER) build --target development -t $(IMAGE_NAME):dev .
	@echo "$(GREEN)开发环境Docker镜像构建完成$(RESET)"

docker-build-prod: ## 构建生产环境Docker镜像
	@echo "$(YELLOW)构建生产环境Docker镜像...$(RESET)"
	$(DOCKER) build --target production -t $(IMAGE_NAME):prod .
	@echo "$(GREEN)生产环境Docker镜像构建完成$(RESET)"

docker-run: ## 运行Docker容器
	@echo "$(YELLOW)运行Docker容器...$(RESET)"
	$(DOCKER) run -p 8002:8002 --name $(PROJECT_NAME) $(IMAGE_NAME):latest

docker-run-dev: ## 运行开发环境Docker容器
	@echo "$(YELLOW)运行开发环境Docker容器...$(RESET)"
	$(DOCKER) run -p 8002:8002 -v $(PWD):/app --name $(PROJECT_NAME)-dev $(IMAGE_NAME):dev

docker-stop: ## 停止Docker容器
	@echo "$(YELLOW)停止Docker容器...$(RESET)"
	$(DOCKER) stop $(PROJECT_NAME) || true
	$(DOCKER) rm $(PROJECT_NAME) || true

# Docker Compose相关
up: ## 启动所有服务
	@echo "$(YELLOW)启动所有服务...$(RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)所有服务已启动$(RESET)"

down: ## 停止所有服务
	@echo "$(YELLOW)停止所有服务...$(RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)所有服务已停止$(RESET)"

logs: ## 查看服务日志
	@echo "$(YELLOW)查看服务日志...$(RESET)"
	$(DOCKER_COMPOSE) logs -f

restart: ## 重启服务
	@echo "$(YELLOW)重启服务...$(RESET)"
	$(DOCKER_COMPOSE) restart analysis-service

# 健康检查和监控
health: ## 检查服务健康状态
	@echo "$(YELLOW)检查服务健康状态...$(RESET)"
	curl -f http://localhost:8002/health || echo "$(RED)服务不可用$(RESET)"

health-detailed: ## 检查详细健康状态
	@echo "$(YELLOW)检查详细健康状态...$(RESET)"
	curl -s http://localhost:8002/health/status | python -m json.tool

check-env: ## 检查环境配置
	@echo "$(YELLOW)检查环境配置...$(RESET)"
	$(PYTHON) scripts/start.py --check

# 数据库相关
db-init: ## 初始化数据库
	@echo "$(YELLOW)初始化数据库...$(RESET)"
	$(PYTHON) scripts/init_db.py

db-migrate: ## 运行数据库迁移
	@echo "$(YELLOW)运行数据库迁移...$(RESET)"
	alembic upgrade head

db-reset: ## 重置数据库
	@echo "$(YELLOW)重置数据库...$(RESET)"
	$(DOCKER_COMPOSE) down postgres
	$(DOCKER) volume rm analysis-service_postgres_data || true
	$(DOCKER_COMPOSE) up -d postgres
	sleep 10
	make db-migrate

# 清理相关
clean: ## 清理临时文件
	@echo "$(YELLOW)清理临时文件...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	@echo "$(GREEN)清理完成$(RESET)"

clean-docker: ## 清理Docker资源
	@echo "$(YELLOW)清理Docker资源...$(RESET)"
	$(DOCKER) system prune -f
	$(DOCKER) volume prune -f
	@echo "$(GREEN)Docker资源清理完成$(RESET)"

# 部署相关
deploy-staging: ## 部署到测试环境
	@echo "$(YELLOW)部署到测试环境...$(RESET)"
	# 这里添加测试环境部署逻辑
	@echo "$(GREEN)测试环境部署完成$(RESET)"

deploy-prod: ## 部署到生产环境
	@echo "$(YELLOW)部署到生产环境...$(RESET)"
	# 这里添加生产环境部署逻辑
	@echo "$(GREEN)生产环境部署完成$(RESET)"

# 监控相关
monitor: ## 启动监控服务
	@echo "$(YELLOW)启动监控服务...$(RESET)"
	$(DOCKER_COMPOSE) up -d prometheus grafana
	@echo "$(GREEN)监控服务已启动$(RESET)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

# 开发工具
shell: ## 进入Python shell
	@echo "$(YELLOW)启动Python shell...$(RESET)"
	$(PYTHON) -c "import sys; sys.path.insert(0, '.'); from app import *; print('Analysis Service Shell Ready')"

notebook: ## 启动Jupyter Notebook
	@echo "$(YELLOW)启动Jupyter Notebook...$(RESET)"
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 文档相关
docs: ## 生成API文档
	@echo "$(YELLOW)生成API文档...$(RESET)"
	$(PYTHON) -c "from main import create_app; import json; app = create_app(); print(json.dumps(app.openapi(), indent=2))" > docs/openapi.json
	@echo "$(GREEN)API文档已生成到 docs/openapi.json$(RESET)"

# 安全检查
security: ## 安全检查
	@echo "$(YELLOW)运行安全检查...$(RESET)"
	bandit -r app/ -f json -o security-report.json || true
	safety check --json --output security-deps.json || true
	@echo "$(GREEN)安全检查完成$(RESET)"

# 性能测试
benchmark: ## 性能基准测试
	@echo "$(YELLOW)运行性能基准测试...$(RESET)"
	# 这里可以添加性能测试逻辑
	@echo "$(GREEN)性能测试完成$(RESET)"

# 完整的CI/CD流程
ci: clean lint format-check test security ## 运行CI流程
	@echo "$(GREEN)CI流程完成$(RESET)"

# 快速启动
quick-start: install up health ## 快速启动完整环境
	@echo "$(GREEN)快速启动完成$(RESET)"
	@echo "服务地址: http://localhost:8002"
	@echo "API文档: http://localhost:8002/docs"
