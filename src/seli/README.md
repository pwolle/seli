# Seli

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

## Roadmap
- [x] Module Meta
- [x] Serialize
- [x] Module repr
- [x] Layers
- [x] Filter jit and filter grad
- [x] Move layers to new param api
- [x] Move tests to new param api
- [x] Optimizer
- [x] Documentation
- [x] CI on github actions
- [x] Push to pypi
- [x] set rngs from module
- [x] Supervised examples
    - [x] Classification
    - [x] Regression

- [ ] Return loss from optimizer
- [ ] Combination layers
- [ ] Unsupervised examples
    - [ ] Autoencoders
    - [ ] VAEs
    - [ ] Normalizing Flows
    - [ ] Diffusion models
    - [ ] Flow matching
    - [ ] Score matching
    - [ ] GANs
- [ ] Convnets
- [ ] Dataloader
- [ ] Datasets
- [ ] Training loop/trainer?
