# LEXAR v1.1.0 — Clean Reinstall & Publishing Guide

## ✅ Step 1: Clean Reinstall (REQUIRED)

The `pyproject.toml` has been updated to:
1. **Remove `torch` from hard dependencies** → Avoids CUDA bloat
2. **Pin `transformers<5.0.0`** → Ensures reproducibility

### Reinstall Commands

```bash
# Navigate to project root
cd /home/garv/projects/legalrag

# Uninstall existing package (cleans editable install)
pip uninstall -y lexar-ai

# Optional: Remove torch/transformers to test clean dependency resolution
pip uninstall -y torch transformers

# Reinstall in editable mode with CPU support
pip install -e .[cpu]

# Or reinstall with all extras
pip install -e .[all]
```

### Verify Installation

```bash
# Test import
python3 test_import.py

# Or quick inline test
python3 -c "from lexar import LexarPipeline, __version__; print(f'✓ LEXAR v{__version__}')"

# Test CLI
lexar --version
```

**Expected output:**
```
✓ LEXAR v1.1.0 imported successfully
✓ LexarPipeline available at: lexar.lexar_pipeline
```

**No CUDA spam expected** if you don't have GPU installed.

---

## 🏗️ Step 2: Build Distribution Package

```bash
# Install build tools (if not already installed)
pip install build twine

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python3 -m build
```

**Output:**
```
dist/
  lexar_ai-1.1.0-py3-none-any.whl
  lexar-ai-1.1.0.tar.gz
```

---

## 🧪 Step 3: Test on TestPyPI (HIGHLY RECOMMENDED)

### 3.1: Upload to TestPyPI

```bash
# Upload to test.pypi.org
python3 -m twine upload --repository testpypi dist/*
```

**You'll be prompted for:**
- Username: `__token__`
- Password: Your TestPyPI API token (get from https://test.pypi.org/manage/account/token/)

### 3.2: Test Installation from TestPyPI

```bash
# Create fresh test environment
python3 -m venv /tmp/lexar_test_env
source /tmp/lexar_test_env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            lexar-ai[cpu]

# Test import
python3 -c "from lexar import LexarPipeline, __version__; print(f'TestPyPI install OK: v{__version__}')"

# Test CLI
lexar --version

# Clean up
deactivate
rm -rf /tmp/lexar_test_env
```

**If this works**, you're ready for PyPI.

---

## 🚀 Step 4: Publish to PyPI (Production)

### 4.1: Upload to PyPI

```bash
# Upload to pypi.org
python3 -m twine upload dist/*
```

**You'll be prompted for:**
- Username: `__token__`
- Password: Your PyPI API token (get from https://pypi.org/manage/account/token/)

### 4.2: Verify Production Installation

```bash
# Fresh environment
python3 -m venv /tmp/lexar_prod_test
source /tmp/lexar_prod_test/bin/activate

# Install from PyPI
pip install lexar-ai[cpu]

# Verify
python3 -c "from lexar import LexarPipeline; print('PyPI install SUCCESS!')"

# Clean up
deactivate
rm -rf /tmp/lexar_prod_test
```

---

## 📋 Step 5: Post-Release Tasks

### 5.1: Create Git Tag

```bash
git add pyproject.toml README.md
git commit -m "Release v1.1.0: Pip-installable package with optional torch"
git tag v1.1.0
git push origin main
git push origin v1.1.0
```

### 5.2: Create GitHub Release

1. Go to https://github.com/yourusername/legalrag/releases
2. Click "Create a new release"
3. Tag: `v1.1.0`
4. Title: `LEXAR v1.1.0 — First Pip-Installable Release`
5. Description:

```markdown
## 🎉 LEXAR v1.1.0 — Pip-Installable Research Framework

**Major Milestone:** LEXAR is now available as a pip-installable package!

### Installation

```bash
pip install lexar-ai[cpu]
```

### What's Included

✅ Evidence-constrained generation with hard attention masking  
✅ Token-level provenance tracking  
✅ Evidence sufficiency gating  
✅ Fine-tuned query encoder (Recall@5: +42.7%)  
✅ CLI support: `lexar --version`  
✅ Clean public API: `from lexar import LexarPipeline`  

### Breaking Changes

- Package moved from `backend.app.services.*` → `lexar.*`
- PyTorch is now an optional dependency (`[cpu]` or `[gpu]` extras)
- Transformers pinned to `<5.0.0` for reproducibility

See [README.md](https://github.com/yourusername/legalrag) for full documentation.
```

### 5.3: Update Package Links

Update `pyproject.toml` URLs once repo is public:

```toml
[project.urls]
Homepage = "https://github.com/ACTUAL_USERNAME/legalrag"
Documentation = "https://github.com/ACTUAL_USERNAME/legalrag#readme"
Repository = "https://github.com/ACTUAL_USERNAME/legalrag"
Issues = "https://github.com/ACTUAL_USERNAME/legalrag/issues"
```

---

## 🔒 Security Best Practices

### API Token Storage

**Option 1: Environment Variables**

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...your-token...
```

**Option 2: `.pypirc` file**

```bash
# Create ~/.pypirc
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-...your-production-token...

[testpypi]
username = __token__
password = pypi-...your-test-token...
EOF

# Secure permissions
chmod 600 ~/.pypirc
```

**NEVER commit tokens to git!**

---

## 📊 Dependency Structure (After Fixes)

### Core Dependencies (Always Installed)

```
transformers>=4.30.0,<5.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
numpy>=1.21.0
pydantic>=2.0.0
```

### Optional Dependencies

```
[cpu]    → torch>=2.0.0 (CPU version)
[gpu]    → torch>=2.0.0 (GPU version)
[server] → fastapi, uvicorn
[dev]    → pytest, black, mypy, etc.
[all]    → Everything above
```

**Why this matters:**
- `sentence-transformers` will auto-install `torch` if needed
- Users can avoid CUDA packages if CPU-only
- Smaller initial install footprint

---

## 🐛 Troubleshooting

### Issue: `transformers 5.x` installed

**Fix:** Uninstall and reinstall with version constraint:
```bash
pip uninstall transformers
pip install "transformers>=4.30.0,<5.0.0"
```

### Issue: CUDA libraries installing on CPU-only system

**Fix:** Install with explicit `[cpu]` extra:
```bash
pip install lexar-ai[cpu]
```

### Issue: Import fails after editable install

**Fix:** Reinstall in editable mode:
```bash
pip uninstall lexar-ai
pip install -e .[cpu]
```

---

## ✅ Final Checklist

Before uploading to PyPI:

- [ ] `pyproject.toml` updated (torch optional, transformers<5.0.0)
- [ ] Clean reinstall tested: `pip install -e .[cpu]`
- [ ] Import works: `python3 -c "from lexar import LexarPipeline"`
- [ ] CLI works: `lexar --version`
- [ ] Built distribution: `python3 -m build`
- [ ] Tested on TestPyPI
- [ ] Installed from TestPyPI successfully
- [ ] Ready to upload to production PyPI

---

**Last Updated:** February 9, 2026  
**Version:** 1.1.0  
**Status:** Ready for PyPI publication
