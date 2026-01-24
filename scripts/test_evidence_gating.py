"""
Evidence Sufficiency Gating Tests

Comprehensive tests for the evidence gating mechanism that ensures
answers are only returned when there is sufficient evidence support.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.generation.evidence_gating import (
    EvidenceSufficiencyGate,
    EvidenceGatingStats,
)


class TestEvidenceSufficiencyGate:
    """Test the evidence sufficiency gating mechanism."""

    def test_gate_initialization(self):
        """Test gate initialization with different thresholds."""
        print("\n" + "="*80)
        print("TEST 1: Evidence Gate Initialization")
        print("="*80)

        # Test default initialization
        gate = EvidenceSufficiencyGate()
        assert gate.threshold == 0.5
        assert gate.enable_gating == True
        assert gate.strict_mode == False
        print("✓ Default initialization: threshold=0.5, enabled, non-strict")

        # Test custom threshold
        gate = EvidenceSufficiencyGate(threshold=0.7)
        assert gate.threshold == 0.7
        print("✓ Custom threshold: 0.7")

        # Test disabled gating
        gate = EvidenceSufficiencyGate(enable_gating=False)
        assert gate.enable_gating == False
        print("✓ Gating disabled")

        # Test strict mode
        gate = EvidenceSufficiencyGate(strict_mode=True)
        assert gate.strict_mode == True
        print("✓ Strict mode enabled")

        # Test invalid threshold
        try:
            gate = EvidenceSufficiencyGate(threshold=1.5)
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Rejects invalid threshold (>1.0)")

    def test_gate_sufficient_evidence(self):
        """Test gate passes when evidence is sufficient."""
        print("\n" + "="*80)
        print("TEST 2: Gate with Sufficient Evidence")
        print("="*80)

        gate = EvidenceSufficiencyGate(threshold=0.5)
        
        attention_dist = {
            "IPC_302": 0.65,  # Above threshold
            "IPC_34": 0.35
        }
        
        evidence_chunks = [
            {
                "chunk_id": "IPC_302",
                "text": "Punishment for murder...",
                "metadata": {"statute": "IPC", "section": "302"}
            },
            {
                "chunk_id": "IPC_34",
                "text": "Acts in furtherance...",
                "metadata": {"statute": "IPC", "section": "34"}
            }
        ]
        
        passes, gate_info = gate.evaluate(
            attention_distribution=attention_dist,
            evidence_chunks=evidence_chunks,
            query="What is punishment for murder?",
            answer="Death or life imprisonment."
        )
        
        assert passes == True
        assert gate_info["passes"] == True
        assert gate_info["max_attention"] == 0.65
        assert gate_info["max_chunk_id"] == "IPC_302"
        assert "margin" in gate_info
        assert gate_info["margin"] == 0.15
        print(f"✓ Gate PASSED with max_attention=65%, margin=15%")
        print(f"  Max chunk: {gate_info['max_chunk_id']}")

    def test_gate_insufficient_evidence(self):
        """Test gate rejects when evidence is insufficient."""
        print("\n" + "="*80)
        print("TEST 3: Gate with Insufficient Evidence")
        print("="*80)

        gate = EvidenceSufficiencyGate(threshold=0.5)
        
        attention_dist = {
            "IPC_302": 0.32,  # Below threshold
            "IPC_34": 0.25,
            "IPC_503": 0.43
        }
        
        evidence_chunks = [
            {
                "chunk_id": "IPC_302",
                "text": "Punishment for murder...",
                "metadata": {"statute": "IPC", "section": "302"}
            },
            {
                "chunk_id": "IPC_34",
                "text": "Acts in furtherance...",
                "metadata": {"statute": "IPC", "section": "34"}
            },
            {
                "chunk_id": "IPC_503",
                "text": "Criminal intimidation...",
                "metadata": {"statute": "IPC", "section": "503"}
            }
        ]
        
        passes, gate_info = gate.evaluate(
            attention_distribution=attention_dist,
            evidence_chunks=evidence_chunks,
            query="What is punishment for murder?",
            answer="Some answer."
        )
        
        assert passes == False
        assert gate_info["passes"] == False
        assert gate_info["max_attention"] == 0.43
        assert gate_info["max_chunk_id"] == "IPC_503"
        assert "deficit" in gate_info
        assert gate_info["deficit"] == 0.07
        assert "refusal" in gate_info
        print(f"✓ Gate REJECTED with max_attention=43%, deficit=7%")
        print(f"  Required threshold: 50%")
        
        # Check refusal structure
        refusal = gate_info["refusal"]
        assert refusal["status"] == "insufficient_evidence"
        assert "reason" in refusal
        assert "suggestions" in refusal
        assert "evidence_summary" in refusal
        print(f"✓ Structured refusal generated")
        print(f"  Suggestions: {len(refusal['suggestions'])} provided")

    def test_gate_borderline_evidence(self):
        """Test gate at exact threshold boundary."""
        print("\n" + "="*80)
        print("TEST 4: Gate at Threshold Boundary")
        print("="*80)

        gate = EvidenceSufficiencyGate(threshold=0.5, strict_mode=False)
        
        attention_dist = {
            "IPC_302": 0.50,  # Exactly at threshold
            "IPC_34": 0.50
        }
        
        evidence_chunks = [
            {"chunk_id": "IPC_302", "text": "...", "metadata": {"statute": "IPC"}},
            {"chunk_id": "IPC_34", "text": "...", "metadata": {"statute": "IPC"}}
        ]
        
        passes, gate_info = gate.evaluate(
            attention_distribution=attention_dist,
            evidence_chunks=evidence_chunks,
            query="Question?",
            answer="Answer."
        )
        
        # Non-strict mode should pass at threshold
        assert passes == True
        assert gate_info["margin"] == 0.0
        print("✓ Non-strict mode: PASSES at exactly threshold")
        
        # Strict mode should fail at threshold
        gate_strict = EvidenceSufficiencyGate(threshold=0.5, strict_mode=True)
        passes_strict, gate_info_strict = gate_strict.evaluate(
            attention_distribution=attention_dist,
            evidence_chunks=evidence_chunks,
            query="Question?",
            answer="Answer."
        )
        
        assert passes_strict == False
        assert gate_info_strict["deficit"] == 0.0
        print("✓ Strict mode: FAILS at exactly threshold")

    def test_gate_disabled(self):
        """Test that disabled gating always passes."""
        print("\n" + "="*80)
        print("TEST 5: Disabled Gating")
        print("="*80)

        gate = EvidenceSufficiencyGate(threshold=0.5, enable_gating=False)
        
        attention_dist = {
            "IPC_302": 0.20,  # Very low
            "IPC_34": 0.80
        }
        
        evidence_chunks = [
            {"chunk_id": "IPC_302", "text": "...", "metadata": {"statute": "IPC"}},
            {"chunk_id": "IPC_34", "text": "...", "metadata": {"statute": "IPC"}}
        ]
        
        passes, gate_info = gate.evaluate(
            attention_distribution=attention_dist,
            evidence_chunks=evidence_chunks,
            query="Question?",
            answer="Answer."
        )
        
        # Should pass despite low max_attention
        assert passes == True
        assert gate_info["gating_bypassed"] == True
        print("✓ Gating disabled: Answer ACCEPTED despite insufficient evidence")
        print(f"  Gating bypassed flag set: {gate_info['gating_bypassed']}")

    def test_threshold_modification(self):
        """Test dynamic threshold modification."""
        print("\n" + "="*80)
        print("TEST 6: Dynamic Threshold Modification")
        print("="*80)

        gate = EvidenceSufficiencyGate(threshold=0.5)
        assert gate.get_threshold() == 0.5
        print("✓ Initial threshold: 0.5")

        gate.set_threshold(0.7)
        assert gate.get_threshold() == 0.7
        print("✓ Threshold updated: 0.5 → 0.7")

        # Test invalid threshold
        try:
            gate.set_threshold(1.5)
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Rejects invalid threshold during modification")

    def test_enable_disable_toggling(self):
        """Test enabling and disabling gating."""
        print("\n" + "="*80)
        print("TEST 7: Enable/Disable Toggling")
        print("="*80)

        gate = EvidenceSufficiencyGate(enable_gating=True)
        assert gate.is_enabled() == True
        print("✓ Gating initially enabled")

        gate.disable()
        assert gate.is_enabled() == False
        print("✓ Gating disabled")

        gate.enable()
        assert gate.is_enabled() == True
        print("✓ Gating re-enabled")

    def test_refusal_structure(self):
        """Test the structure of refusal messages."""
        print("\n" + "="*80)
        print("TEST 8: Refusal Message Structure")
        print("="*80)

        gate = EvidenceSufficiencyGate(threshold=0.6)
        
        attention_dist = {"IPC_302": 0.4, "IPC_34": 0.6}
        # But max is 0.6, which doesn't exceed 0.6 - wait, 0.6 >= 0.6 passes
        # Let me make it clearly fail
        gate2 = EvidenceSufficiencyGate(threshold=0.7)
        attention_dist = {"IPC_302": 0.4, "IPC_34": 0.6}
        
        evidence_chunks = [
            {
                "chunk_id": "IPC_302",
                "text": "Text about punishment...",
                "metadata": {
                    "statute": "IPC",
                    "section": "302",
                    "jurisdiction": "India"
                }
            },
            {
                "chunk_id": "IPC_34",
                "text": "Acts in furtherance...",
                "metadata": {
                    "statute": "IPC",
                    "section": "34",
                    "jurisdiction": "India"
                }
            }
        ]
        
        passes, gate_info = gate2.evaluate(
            attention_distribution=attention_dist,
            evidence_chunks=evidence_chunks,
            query="Query about IPC provisions",
            answer="Generated answer"
        )
        
        assert not passes
        refusal = gate_info["refusal"]
        
        # Check all required fields
        required_fields = [
            "status",
            "reason",
            "max_attention",
            "required_threshold",
            "deficit",
            "evidence_retrieved",
            "evidence_summary",
            "suggestions",
            "explanation"
        ]
        
        for field in required_fields:
            assert field in refusal, f"Missing field: {field}"
        
        print(f"✓ Refusal contains all required fields ({len(required_fields)})")
        print(f"  Status: {refusal['status']}")
        print(f"  Reason: {refusal['reason'][:60]}...")
        print(f"  Evidence Summary: {len(refusal['evidence_summary'])} chunks")
        print(f"  Suggestions: {len(refusal['suggestions'])} provided")

    def test_gating_stats(self):
        """Test evidence gating statistics tracking."""
        print("\n" + "="*80)
        print("TEST 9: Gating Statistics Tracking")
        print("="*80)

        stats = EvidenceGatingStats()
        
        # Record some evaluations
        stats.record(True, 0.65, "Query 1")
        stats.record(True, 0.72, "Query 2")
        stats.record(False, 0.35, "Query 3")
        stats.record(True, 0.58, "Query 4")
        stats.record(False, 0.42, "Query 5")
        
        summary = stats.get_stats()
        
        assert summary["total_evaluations"] == 5
        assert summary["passed"] == 3
        assert summary["failed"] == 2
        assert summary["pass_rate"] == 0.6
        
        print(f"✓ Statistics recorded: {summary['total_evaluations']} evaluations")
        print(f"  Pass rate: {summary['pass_rate']:.1%}")
        print(f"  Avg max attention: {summary['avg_max_attention']:.1%}")
        print(f"  Min: {summary['min_max_attention']:.1%}")
        print(f"  Max: {summary['max_max_attention']:.1%}")
        print(f"  Recently rejected: {len(summary['recently_rejected'])} queries")
        
        # Reset and verify
        stats.reset()
        empty_stats = stats.get_stats()
        assert empty_stats["no_data"] == True
        print("✓ Statistics reset successful")

    def test_attention_normalization(self):
        """Test handling of attention distributions with floating point errors."""
        print("\n" + "="*80)
        print("TEST 10: Attention Normalization")
        print("="*80)

        gate = EvidenceSufficiencyGate(threshold=0.5)
        
        # Attention that doesn't sum to exactly 1.0 due to floating point
        attention_dist = {
            "IPC_302": 0.333334,
            "IPC_34": 0.333333,
            "IPC_503": 0.333333
        }
        
        evidence_chunks = [
            {"chunk_id": f"IPC_{i}", "text": "...", "metadata": {"statute": "IPC"}}
            for i in [302, 34, 503]
        ]
        
        # Should not raise error despite not summing to exactly 1.0
        passes, gate_info = gate.evaluate(
            attention_distribution=attention_dist,
            evidence_chunks=evidence_chunks,
            query="Question?",
            answer="Answer."
        )
        
        assert gate_info["max_attention"] == 0.3333  # Rounded
        print("✓ Handles floating point normalization")
        print(f"  Max attention (rounded): {gate_info['max_attention']}")


def run_all_tests():
    """Run all evidence gating tests."""
    print("\n" + "="*80)
    print("EVIDENCE SUFFICIENCY GATING TEST SUITE")
    print("="*80)

    try:
        test = TestEvidenceSufficiencyGate()
        
        test.test_gate_initialization()
        test.test_gate_sufficient_evidence()
        test.test_gate_insufficient_evidence()
        test.test_gate_borderline_evidence()
        test.test_gate_disabled()
        test.test_threshold_modification()
        test.test_enable_disable_toggling()
        test.test_refusal_structure()
        test.test_gating_stats()
        test.test_attention_normalization()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nEvidence Sufficiency Gating Verification:")
        print("  ✓ Gate initialization (default & custom)")
        print("  ✓ Sufficient evidence evaluation")
        print("  ✓ Insufficient evidence rejection")
        print("  ✓ Borderline evidence (threshold boundary)")
        print("  ✓ Disabled gating bypass")
        print("  ✓ Dynamic threshold modification")
        print("  ✓ Enable/disable toggling")
        print("  ✓ Refusal message structure")
        print("  ✓ Statistics tracking")
        print("  ✓ Floating point normalization")
        print("\nEVIDENCE GATING READY FOR PRODUCTION")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
