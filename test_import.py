#!/usr/bin/env python3
"""Test LEXAR package imports after reinstall."""

print("Testing LEXAR package installation...")
print("=" * 60)

try:
    from lexar import LexarPipeline, __version__
    print(f"✓ LEXAR v{__version__} imported successfully")
    print(f"✓ LexarPipeline available at: {LexarPipeline.__module__}")
    print(f"✓ LexarPipeline class: {LexarPipeline}")
    print()
    print("=" * 60)
    print("SUCCESS: Package installation verified!")
    print("=" * 60)
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
