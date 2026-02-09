# LEXAR v1.1.0 Release Notes

## 🎉 Major Milestone: PyPI Package Release

LEXAR is now available as a pip-installable package on PyPI! Install with a single command and integrate legal RAG capabilities into your projects.

## 📦 Installation

```bash
# CPU-only installation (recommended for most users)
pip install lexar-ai[cpu]

# GPU-accelerated installation
pip install lexar-ai[gpu]

# Development installation with all extras
pip install lexar-ai[all]
```

**PyPI Package**: https://pypi.org/project/lexar-ai/1.1.0/

## 🚀 What's New

### Package Distribution
- **pip-installable framework** - LEXAR is now a proper Python package
- **Clean public API** - Simple imports: `from lexar import LexarPipeline`
- **Optional dependencies** - Choose CPU or GPU installation based on your needs
- **CLI tool** - Command-line interface: `lexar --version`, `lexar query`, `lexar info`

### Dependency Optimization
- **PyTorch made optional** - Not forced on all users; available via `[cpu]` and `[gpu]` extras
- **Transformers pinned to <5.0.0** - Ensures reproducibility and stability
- **Minimal core dependencies** - Only essential packages in base installation

### Developer Experience
- **PEP 621 packaging** - Modern `pyproject.toml` configuration
- **Multiple installation modes** - `[cpu]`, `[gpu]`, `[dev]`, `[server]`, `[all]`
- **Editable installs** - `pip install -e .[cpu]` for local development
- **Entry point scripts** - CLI accessible system-wide after install

## 📋 Features

### Core Capabilities
- **Multi-stage retrieval** - FAISS-based vector search with BM25 fallback
- **Neural reranking** - Cross-encoder reranking for precision
- **Legal-aware chunking** - Optimized for Indian legal documents (IPC, judgments, statutes)
- **Citation rendering** - Proper legal citation formatting
- **Evidence-constrained generation** - Grounded responses with provenance
- **Token-level provenance** - Track which documents influenced each token

### Supported Document Types
- Indian Penal Code (IPC) sections
- Supreme Court judgments
- High Court judgments  
- Indian statutes and acts
- Legal commentaries

## 🛠️ Technical Details

### System Requirements
- **Python**: 3.8+
- **OS**: Linux, macOS, Windows
- **RAM**: 8GB minimum (16GB recommended for large corpora)
- **Disk**: 2GB for models and indices

### Core Dependencies
```
transformers>=4.30.0,<5.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
numpy>=1.21.0
pydantic>=2.0.0
```

### Optional Dependencies
- **CPU**: `torch>=2.0.0` (CPU-only)
- **GPU**: `torch>=2.0.0` with CUDA support
- **Server**: `fastapi`, `uvicorn`, `python-multipart`
- **Development**: `pytest`, `black`, `ruff`

## 📖 Quick Start

### Basic Usage

```python
from lexar import LexarPipeline

# Initialize pipeline
pipeline = LexarPipeline(
    index_path="data/faiss_index/ipc.index",
    chunk_ids_path="data/faiss_index/ipc_chunk_ids.json"
)

# Query the system
result = pipeline.query(
    query="What is the punishment for theft under IPC?",
    top_k=5
)

print(result.answer)
print(result.citations)
```

### CLI Usage

```bash
# Check version
lexar --version

# Run interactive query
lexar query "What is the punishment for theft?"

# Show system info
lexar info
```

## 🔄 Migration Guide

If you were using LEXAR from the source repository:

**Before (source install)**:
```python
from backend.app.services.lexar_pipeline import LexarPipeline
```

**After (pip install)**:
```python
from lexar import LexarPipeline
```

## 🐛 Bug Fixes

- Fixed dependency resolution issues with PyTorch installations
- Resolved import path inconsistencies
- Corrected CLI entry point registration

## 📚 Documentation

- **README.md** - Updated with pip installation instructions
- **RELEASE_GUIDE.md** - Step-by-step release workflow
- **PUBLISHING.md** - PyPI publishing guide
- **Package documentation** - Available at https://pypi.org/project/lexar-ai/

## 🔗 Links

- **PyPI Package**: https://pypi.org/project/lexar-ai/1.1.0/
- **TestPyPI Package**: https://test.pypi.org/project/lexar-ai/1.1.0/
- **Source Code**: https://github.com/yourusername/legalrag
- **Documentation**: [Project README](README.md)

## 🙏 Acknowledgments

This release marks LEXAR's transition from a research prototype to a production-ready package. Special thanks to the open-source community for their contributions to the underlying libraries.

## 📝 Changelog Summary

### Added
- PyPI package distribution (`lexar-ai`)
- Public API with `LexarPipeline` export
- CLI tool with `lexar` command
- Optional dependency groups `[cpu]`, `[gpu]`, `[dev]`, `[server]`, `[all]`
- Version metadata via `__version__.py`
- PEP 621 compliant packaging

### Changed
- Import paths: `backend.app.services.*` → `lexar.*`
- PyTorch moved to optional dependencies
- Transformers pinned to `<5.0.0` for stability

### Fixed
- Dependency resolution in fresh environments
- CLI entry point registration
- Import path inconsistencies

---

**Full Package Name**: `lexar-ai`  
**Version**: 1.1.0  
**Release Date**: February 9, 2026  
**License**: [Your License]  
**Maintainer**: [Your Name/Organization]
