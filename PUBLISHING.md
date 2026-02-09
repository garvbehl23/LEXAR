# Publishing LEXAR to PyPI

This guide explains how to publish the `lexar-ai` package to PyPI.

## Prerequisites

1. Install build tools:
```bash
pip install build twine
```

2. Create PyPI account:
   - Visit https://pypi.org/account/register/
   - Verify your email

3. Create API token:
   - Go to https://pypi.org/manage/account/token/
   - Create token with scope: "Entire account"
   - Save the token securely

## Build the Package

```bash
# Navigate to repository root
cd /home/garv/projects/legalrag

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution
python3 -m build
```

This creates:
- `dist/lexar_ai-1.1.0-py3-none-any.whl` (wheel)
- `dist/lexar-ai-1.1.0.tar.gz` (source distribution)

## Test the Package Locally

```bash
# Install in a fresh virtual environment
python3 -m venv test_env
source test_env/bin/activate

# Install from local wheel
pip install dist/lexar_ai-1.1.0-py3-none-any.whl

# Test import
python3 -c "from lexar import LexarPipeline, __version__; print(__version__)"

# Test CLI
lexar --version

# Clean up
deactivate
rm -rf test_env
```

## Upload to Test PyPI (Recommended First)

```bash
# Upload to test.pypi.org
python3 -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ lexar-ai
```

## Upload to PyPI (Production)

```bash
# Upload to pypi.org
python3 -m twine upload dist/*

# Verify upload
pip install lexar-ai
python3 -c "from lexar import LexarPipeline; print('Success!')"
```

## Post-Release

1. Create Git tag:
```bash
git tag v1.1.0
git push origin v1.1.0
```

2. Create GitHub release:
   - Go to https://github.com/yourusername/legalrag/releases
   - Click "Create a new release"
   - Tag: v1.1.0
   - Title: "LEXAR v1.1.0 - First Pip-Installable Release"
   - Description: See CHANGELOG.md

3. Update documentation:
   - Add installation instructions
   - Update version references
   - Add migration guide if needed

## Version Bumping

To release a new version:

1. Update version in `lexar/__version__.py`:
```python
__version__ = "1.2.0"
```

2. Update version in `pyproject.toml`:
```toml
[project]
version = "1.2.0"
```

3. Build and upload as above

## Troubleshooting

### Import errors after installation
- Ensure `__init__.py` files exist in all package directories
- Check `pyproject.toml` `packages.find` configuration

### Missing dependencies
- Add missing packages to `dependencies` in `pyproject.toml`
- Test in clean environment

### CLI not working
- Verify `[project.scripts]` in `pyproject.toml`
- Check `cli.py` has `main()` function

## Security

- Never commit API tokens to Git
- Use `.pypirc` for credentials:
```ini
[pypi]
username = __token__
password = pypi-...your-token...
```

- Protect `.pypirc`:
```bash
chmod 600 ~/.pypirc
```
