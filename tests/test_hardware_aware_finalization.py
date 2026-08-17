from __future__ import annotations

import copy
import unittest

from scripts.audit_hardware_aware_campaign import (
    FinalAuditError,
    PRE_AUDIT_STATE,
    PRIMARY_OUTCOME,
    validate_independent_evidence,
    validate_scope_and_leakage,
    validate_terminal_inputs,
)
from scripts.finalize_hardware_aware_campaign import pareto_frontier


class HardwareAwareFinalizationTests(unittest.TestCase):
    @staticmethod
    def terminal_fixture():
        phase_names = (
            "phase0", "instrumentation", "runtime_attribution", "runtime_repairs",
            "zero_tail_quality", "decoder_timing", "early_compression", "analysis",
        )
        frontier_names = {
            "acc_vs_natural_end_to_end", "miou_vs_natural_end_to_end", "acc_vs_prefill",
            "miou_vs_prefill", "acc_vs_ttft", "miou_vs_ttft", "acc_vs_prompt_kv",
            "miou_vs_prompt_kv", "acc_vs_final_kv", "miou_vs_final_kv",
            "acc_vs_token_layer_work", "miou_vs_token_layer_work",
            "acc_vs_final_visual_tokens", "miou_vs_final_visual_tokens",
        }
        registry = {
            "phases": {name: {"status": "complete"} for name in phase_names},
            "status": PRE_AUDIT_STATE,
            "primary_outcome": PRIMARY_OUTCOME,
        }
        registry["phases"]["final_audit"] = {"status": "pending"}
        controller = {
            "state": PRE_AUDIT_STATE,
            "current_phase": "final_audit",
            "active_workers": [],
            "primary_outcome": PRIMARY_OUTCOME,
        }
        strict = {"strict_winners": [], "phase5_decision": "TRIGGERED_NO_DECODER_INTERNAL_STRICT_WIN"}
        branches = {
            "phase5_terminal": True,
            "phase5_result": "NO_QUALITY_FEASIBLE_EARLY_COMPRESSION_CANDIDATE",
            "decisions": {
                "E196_POOL": {"decision": "SKIPPED_CONSERVATIVE_POOL_NOT_QUALITY_FEASIBLE"},
                "E256_LORA": {"decision": "SKIPPED_LORA_NEAR_QUALITY_GATE_FAILED"},
                "EARLY_COMPRESSION_MATCHED_TIMING": {"decision": "SKIPPED_NO_QUALITY_FEASIBLE_EARLY_CANDIDATE"},
                "EARLY_COMPRESSION_FIXED32": {"decision": "SKIPPED_NO_QUALITY_FEASIBLE_EARLY_CANDIDATE"},
            },
        }
        decision = {"primary_outcome": PRIMARY_OUTCOME, "single_recommended_method": "K12_ORIGINAL"}
        frontiers = {"frontiers": {name: {} for name in frontier_names}}
        report = f"Primary outcome: `{PRIMARY_OUTCOME}`.\n\nCampaign state before independent audit: `{PRE_AUDIT_STATE}`.\n"
        return registry, controller, strict, branches, decision, frontiers, report

    def test_pareto_frontier_keeps_tradeoffs_and_removes_dominated_point(self):
        records = [
            {"method": "fast", "quality": 0.40, "cost": 1.0},
            {"method": "accurate", "quality": 0.50, "cost": 2.0},
            {"method": "dominated", "quality": 0.39, "cost": 3.0},
        ]
        self.assertEqual(pareto_frontier(records, "quality", "cost"), ["fast", "accurate"])

    def test_terminal_audit_fixture_passes(self):
        result = validate_terminal_inputs(*self.terminal_fixture())
        self.assertEqual(result["strict_winner_count"], 0)
        self.assertEqual(result["recommendation"], "K12_ORIGINAL")

    def test_terminal_audit_rejects_active_controller(self):
        fixture = list(self.terminal_fixture())
        fixture[1] = copy.deepcopy(fixture[1])
        fixture[1]["active_workers"] = [{"pid": 7}]
        with self.assertRaisesRegex(FinalAuditError, "active workers"):
            validate_terminal_inputs(*fixture)

    def test_terminal_audit_rejects_incorrect_branch_skip(self):
        fixture = list(self.terminal_fixture())
        fixture[3] = copy.deepcopy(fixture[3])
        fixture[3]["decisions"]["E256_LORA"]["decision"] = "complete"
        with self.assertRaisesRegex(FinalAuditError, "E256_LORA"):
            validate_terminal_inputs(*fixture)

    def test_terminal_audit_rejects_missing_pareto_axis(self):
        fixture = list(self.terminal_fixture())
        fixture[5] = copy.deepcopy(fixture[5])
        del fixture[5]["frontiers"]["miou_vs_final_visual_tokens"]
        with self.assertRaisesRegex(FinalAuditError, "Pareto"):
            validate_terminal_inputs(*fixture)

    def test_scope_audit_rejects_any_lora_attempt(self):
        registry = {
            "cells": {
                "E256_LORA": {
                    "attempts": ["early_compression/training/e256_lora_attempt1"],
                    "status": "SKIPPED_LORA_NEAR_QUALITY_GATE_FAILED",
                }
            }
        }
        with self.assertRaisesRegex(FinalAuditError, "LoRA attempt"):
            validate_scope_and_leakage(registry)

    def test_accepted_zero_tail_is_live_not_oracle_only(self):
        evidence = validate_independent_evidence()
        self.assertTrue(evidence["all_independently_validated"])
        self.assertEqual(evidence["zero_tail_quality_runs"], 2)


if __name__ == "__main__":
    unittest.main()
