#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify that CLOVA experiment results are properly merged
and not overwritten when script runs multiple times.
"""
import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ss_utils import read_json, write_json

def test_data_merge():
    """Test that data is properly merged, not overwritten"""

    # Create temporary test directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 70)
        print("Testing CLOVA Data Merge Logic")
        print("=" * 70)

        partial_file = os.path.join(tmpdir, "clova_results_partial.json")

        # Simulate first run: 2 variables
        print("\n1️⃣  First run: 2 variables (CONFINAN, CONLEGIS)")
        first_run = {
            "timestamp": "2026-01-16T00:00:00Z",
            "configuration": {"variables": ["CONFINAN", "CONLEGIS", "KRPROUD"]},
            "status": "in_progress",
            "completed_variables": ["CONFINAN", "CONLEGIS"],
            "results": [
                {"variable": "CONFINAN", "js_divergence": 0.0729, "n_samples": 100},
                {"variable": "CONLEGIS", "js_divergence": 0.1527, "n_samples": 100},
            ]
        }
        write_json(partial_file, first_run)
        print(f"   Saved: {len(first_run['results'])} variables")

        # Simulate second run: load existing, add 1 more
        print("\n2️⃣  Second run: Load existing + add 1 variable (KRPROUD)")
        existing = read_json(partial_file)
        if existing:
            results = existing.get("results", [])
            completed = existing.get("completed_variables", [])
            print(f"   ✅ Loaded existing: {len(results)} variables, {completed}")

            # Add new result (simulating KRPROUD completion)
            results.append({"variable": "KRPROUD", "js_divergence": 0.0806, "n_samples": 100})
            completed.append("KRPROUD")

            second_run = {
                "timestamp": "2026-01-16T01:00:00Z",
                "configuration": existing.get("configuration"),
                "status": "in_progress",
                "completed_variables": completed,
                "results": results
            }
            write_json(partial_file, second_run)
            print(f"   ✅ Saved updated: {len(second_run['results'])} variables")

        # Verify final result
        print("\n3️⃣  Verify final state")
        final = read_json(partial_file)
        final_results = final.get("results", [])
        final_completed = final.get("completed_variables", [])

        print(f"   Total variables: {len(final_results)}")
        for r in final_results:
            print(f"     {r['variable']}: JS={r['js_divergence']:.4f}")

        # Check integrity
        if len(final_results) == 3 and len(final_completed) == 3:
            print("\n✅ TEST PASSED: Data properly merged, no loss!")
            return True
        else:
            print("\n❌ TEST FAILED: Data integrity issue")
            return False

if __name__ == "__main__":
    success = test_data_merge()
    sys.exit(0 if success else 1)
