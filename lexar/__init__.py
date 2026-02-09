"""
LEXAR: Legal EXplainable Augmented Reasoner

A retrieval-augmented generation system for legal question answering with 
strict evidence grounding and explainable provenance.

Public API:
    from lexar import LexarPipeline

Basic Usage:
    pipeline = LexarPipeline()
    result = pipeline.answer("What is the punishment for murder under IPC?")
    print(result["answer"])

For detailed documentation, see: https://github.com/yourusername/legalrag
"""

from lexar.__version__ import __version__
from lexar.lexar_pipeline import LexarPipeline

__all__ = [
    "LexarPipeline",
    "__version__",
]
