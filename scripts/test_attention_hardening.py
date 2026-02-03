"""
Test that attention weight capture fails loudly when it cannot capture weights.

This validates that the hardening is working as expected.
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.app.services.generation.lexar_generator import LexarGenerator
from backend.app.services.generation.token_provenance import TokenProvenanceTracker


def test_provenance_fails_without_attention():
    """Test that provenance computation raises error without attention weights."""
    
    # Create a tracker with no attention weights
    tracker = TokenProvenanceTracker(
        token_ids_to_chunk_ids={0: "chunk1", 1: "chunk2"},
        enable_tracking=True
    )
    
    # Record some tokens
    tracker.record_token("hello")
    tracker.record_token("world")
    
    # Try to compute provenance without attention weights
    try:
        tracker.compute_provenances()
        print("❌ FAIL: Should have raised RuntimeError")
        return False
    except RuntimeError as e:
        if "No attention weights recorded" in str(e):
            print(f"✅ PASS: Correctly raised error: {e}")
            return True
        else:
            print(f"❌ FAIL: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Wrong exception type: {type(e).__name__}: {e}")
        return False


def test_hook_registration_validation():
    """Test that hook registration fails loudly if it cannot register hooks."""
    
    # Create a generator with real model to test hook registration
    print("\n\nTesting hook registration on real model...")
    generator = LexarGenerator()
    
    # Clear any existing hooks
    generator._remove_attention_hooks()
    generator._clear_attention_capture()
    
    # Try to register hooks
    try:
        generator._register_attention_hooks()
        hooks_count = generator._hooks_registered_count
        print(f"✅ PASS: Successfully registered {hooks_count} hooks")
        
        # Clean up
        generator._remove_attention_hooks()
        return True
    except RuntimeError as e:
        print(f"❌ FAIL: Hook registration failed (this may be OK if model structure changed): {e}")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Attention Weight Capture Hardening")
    print("=" * 80)
    
    print("\n1. Testing provenance without attention weights...")
    test1 = test_provenance_fails_without_attention()
    
    print("\n2. Testing hook registration validation...")
    test2 = test_hook_registration_validation()
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Test 1 (Fail without attention): {'PASS' if test1 else 'FAIL'}")
    print(f"  Test 2 (Hook registration): {'PASS' if test2 else 'FAIL'}")
    
    if test1 and test2:
        print("\n✅ All tests passed! Attention capture hardening is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
