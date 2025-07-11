# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Development Environment
```bash
# Start development server (hot reload)
python start.py
# or
python main.py --reload

# Start production server
python main.py --env production --workers 4

# Quick start with Makefile
make dev          # Development mode
make prod         # Production mode
make quick-start  # Install deps + start services
```

### Testing
```bash
# Run all tests with coverage
make test
# or
pytest tests/ -v --cov=app --cov-report=html

# Run specific test types
make test-unit           # Unit tests only
make test-integration    # Integration tests only
pytest -m "unit"         # Using markers
pytest -m "integration"
```

### Code Quality
```bash
# Lint and format code
make lint          # Run flake8, pylint, mypy
make format        # Format with black + isort
make format-check  # Check formatting only

# Individual tools
flake8 app/ --max-line-length=120
black app/ --line-length=120
isort app/ --profile=black
mypy app/ --ignore-missing-imports
```

### Docker Operations
```bash
# Using Docker Compose (recommended)
make up            # Start all services
make down          # Stop all services
make logs          # View logs

# Using Docker directly
make docker-build  # Build image
make docker-run    # Run container
```

## High-Level Architecture

### Application Structure
This is a **FastAPI-based video analysis service** with a sophisticated plugin architecture:

```
app/
├── core/          # Core systems (analyzer, memory, storage)
├── controllers/   # API endpoints and request handling
├── services/      # Business logic layer
├── plugins/       # Plugin system for extensibility
├── models/        # Pydantic models for data validation
├── repositories/  # Data access layer
└── factory.py     # Application factory for different environments
```

### Key Architectural Patterns

#### 1. Plugin Architecture
- **Dynamic Analyzer Loading**: Analyzers are loaded as plugins with automatic registration
- **Registry Pattern**: `AnalyzerRegistry` manages analyzer discovery and instantiation
- **Decorator-based Registration**: Use `@register_analyzer` for automatic plugin registration
- **Hot Reload Support**: Runtime plugin updates without service restart

#### 2. Zero-Copy Memory Management
- **Memory Pool Design**: Pre-allocated buffer pools prevent runtime allocation
- **Reference Counting**: Automatic cleanup when frame references drop to zero
- **TimeAxis Management**: Temporal frame buffering for video analysis
- **Pool Recycling**: Efficient memory reuse for performance

#### 3. Factory Pattern Implementation
- **ApplicationFactory**: Creates FastAPI apps with different configurations
- **AnalyzerFactory**: Dynamic analyzer creation based on model codes
- **Service Factory**: Dependency injection for service layer

#### 4. Result Processing Pipeline
- **Modular Processors**: Pluggable processors for different output types
- **Async Processing**: Non-blocking result handling
- **Error Resilience**: Individual processor failures don't affect others

### Core Systems

#### Analyzer System
- **Base Interface**: All analyzers implement `BaseAnalyzer` with fallback compatibility
- **Multiple Call Patterns**: Supports `analyze_frame()`, `detect()`, `process_frame()` for different analyzer types
- **FPS Control**: Intelligent frame rate control with automatic dropping for performance
- **Model Management**: Integrated model loading and validation

#### Memory Management
- **Zero-Copy Operations**: Direct memory access without copying data
- **Frame Buffer**: Reference-counted frame storage with metadata
- **Memory Monitoring**: Real-time usage tracking with cleanup triggers
- **Pool Management**: Pre-allocated buffers to avoid runtime allocation

#### Storage System
- **Hierarchical Structure**: Organized storage with automatic directory creation
- **Cache Management**: Intelligent cache cleanup and optimization
- **Resource Monitoring**: Storage usage tracking and alerts
- **Backup System**: Automated backup with compression

#### Configuration Management
- **Pydantic Settings**: Modern configuration with environment-based settings
- **YAML Configuration**: 
  - `analyzer_plugins.yaml` - Plugin configurations
  - `analyzer_configs.yaml` - Analyzer presets
- **Environment Support**: Development, testing, production configurations

### Service Layer Patterns
- **BaseService**: Common functionality with dependency injection
- **Repository Pattern**: Data access abstraction
- **Async Support**: Full async/await support throughout
- **Error Handling**: Unified exception handling with business logic separation

### API Design
- **RESTful Controllers**: Standard response formats and error handling
- **Pydantic Validation**: Request/response validation with models
- **Dependency Injection**: Service layer integration
- **Logging**: Structured logging with request tracking

## Key Development Guidelines

### Adding New Analyzers
1. Create analyzer class inheriting from `BaseAnalyzer`
2. Use `@register_analyzer` decorator for automatic registration
3. Implement required methods: `analyze_frame()`, `analyze_batch()`
4. Add configuration to `analyzer_plugins.yaml`
5. Test with both unit and integration tests

### Memory Management
- Use `MemoryPool` for zero-copy operations
- Implement reference counting for custom frame types
- Monitor memory usage in analysis loops
- Use `TimeAxis` for temporal frame management

### Error Handling
- Use custom exceptions from `app.exceptions`
- Implement proper error handling in controllers
- Log errors with structured logging
- Use business exceptions for domain logic errors

### Testing Strategy
- Unit tests for individual components
- Integration tests for service interactions
- Use pytest markers for test categorization
- Mock external dependencies in tests

### Configuration
- Use environment variables for deployment-specific settings
- Store analyzer configurations in YAML files
- Implement validation for configuration schemas
- Support multiple deployment environments

## Development Notes

### Performance Considerations
- The zero-copy memory architecture is critical for real-time video analysis
- Use async/await patterns for I/O operations
- Pool reuse is essential for memory efficiency
- Monitor memory usage during development

### Plugin Development
- Plugins are automatically discovered and registered
- Use the registry pattern for type-safe plugin management
- Support hot reload for development convenience
- Implement proper dependency resolution

### Service Integration
- Services use dependency injection for loose coupling
- Implement caching where appropriate
- Use repositories for data access abstraction
- Follow async patterns throughout the service layer

## Troubleshooting

### Common Issues
- **Memory Leaks**: Check reference counting in custom analyzers
- **Plugin Loading**: Verify plugin registration and dependencies
- **Performance**: Monitor memory pool usage and frame buffer recycling
- **Configuration**: Validate YAML syntax and schema compliance

### Development Environment
- Ensure Python 3.8+ is installed
- Install requirements: `pip install -r requirements.txt`
- Install dev dependencies: `pip install -r requirements-dev.txt`
- Use virtual environment for isolation

### Service Dependencies
- Redis for caching (optional but recommended)
- Various AI model files in `storage/models/`
- Proper directory structure in `storage/`