# Seli

A Python package called seli.

## Installation

You can install the package via pip:

```bash
pip install seli
```

## Usage

```python
import seli

# Example usage
result = seli.example_function()
print(result)
```

## Development

### Setup

1. Clone the repository
2. Install development dependencies:

```bash
pip install -e ".[dev]"
```

3. Set up pre-commit hooks:

```bash
pre-commit install
```

### Testing

Run tests with pytest:

```bash
pytest
```

### Code Quality

Ruff is configured to run automatically before each commit via pre-commit hooks. You can also run it manually:

```bash
# Check for issues
ruff check .

# Fix issues automatically
ruff check --fix .

# Format code
ruff format .
```

### Documentation

Build the documentation:

```bash
pip install -e ".[docs]"
cd docs
make html
```

The documentation will be available in `docs/_build/html`.


### Roadmap
- [x] Module Meta
- [x] Serialize
- [ ] Layers
- [ ] Optimizer
- [ ] Dataloader
- [ ] Supervised examples
- [ ] Unsupervised examples
