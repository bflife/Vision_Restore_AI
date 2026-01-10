# Contributing to Vision-Restore AI

Thank you for your interest in contributing to Vision-Restore AI! We welcome contributions from the community and are grateful for any help you can provide.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

### Our Standards

- Be welcoming and inclusive
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Finding Issues

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are open for contribution
- Feel free to ask questions on any issue before starting work

## Development Setup

### 1. Fork the Repository

Click the "Fork" button on the [repository page](https://github.com/Rav-xyl/vision-restore-ai).

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/vision-restore-ai.git
cd vision-restore-ai
```

### 3. Create a Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 4. Install Development Dependencies

```bash
pip install -e ".[dev,gpu]"
```

### 5. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Branch Naming Conventions

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Format Your Code

```bash
# Format with black
black vision_restore/

# Check with ruff
ruff check vision_restore/
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new upscaling algorithm"
```

#### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Test additions or modifications
- `chore:` - Build process or auxiliary tool changes

## Submitting a Pull Request

### 1. Push Your Branch

```bash
git push origin feature/your-feature-name
```

### 2. Open a Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request"
3. Fill in the PR template
4. Link related issues

### PR Checklist

- [ ] Code follows the project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventions
- [ ] PR description clearly explains changes

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Code Organization

```python
"""
Module docstring explaining the purpose.
"""

# Standard library imports
import os
from pathlib import Path

# Third-party imports
import numpy as np
import cv2

# Local imports
from vision_restore.core.config import Config

# Constants
MAX_TILE_SIZE = 512

# Classes and functions follow...
```

### Documentation

- All public functions must have docstrings
- Use Google-style docstrings:

```python
def process_image(image: np.ndarray, scale: int = 4) -> np.ndarray:
    """Process an image with upscaling.
    
    Args:
        image: Input image as BGR numpy array
        scale: Upscaling factor (2, 4, or 8)
        
    Returns:
        Upscaled image as numpy array
        
    Raises:
        ValueError: If scale is not 2, 4, or 8
    """
```

## Testing Guidelines

### Test Structure

```
tests/
├── __init__.py
├── test_engine.py      # Engine tests
├── test_tiling.py      # Tiling tests
├── test_gui.py         # GUI component tests
└── conftest.py         # Shared fixtures
```

### Writing Tests

```python
import pytest
import numpy as np

class TestTilingEngine:
    """Tests for the TilingEngine class."""
    
    def test_should_tile_large_image(self):
        """Test that large images are detected for tiling."""
        from vision_restore.engine.tiling import TilingEngine
        
        engine = TilingEngine(tile_size=256)
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        
        assert engine.should_tile(image) is True
```

### Running Specific Tests

```bash
# Run a specific test file
pytest tests/test_engine.py -v

# Run a specific test
pytest tests/test_engine.py::TestTilingEngine::test_should_tile_large_image -v

# Run with coverage
pytest tests/ --cov=vision_restore --cov-report=html
```

## Documentation

### Types of Documentation

1. **Code Comments**: Explain complex logic
2. **Docstrings**: Document all public APIs
3. **README.md**: Project overview and quick start
4. **docs/**: In-depth documentation (if applicable)

### Building Documentation

```bash
# Generate API docs (if using Sphinx)
cd docs
make html
```

## Need Help?

- Open an issue for questions
- Join discussions in existing issues
- Tag maintainers for blocked PRs

---

Thank you for contributing to Vision-Restore AI! 🎉
