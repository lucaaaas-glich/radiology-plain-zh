"""Metadata-only historical inventory; no model or external service is needed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_exposure_manifest import Inventory, fingerprint, main


class ExposureManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, path, rows):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_normalized_hash_matches_existing_contract(self):
        self.assertEqual(fingerprint(" A\n\tB  STRASSE "), fingerprint("a b Straße"))
        self.assertNotEqual(fingerprint("right lung"), fingerprint("left lung"))

    def test_same_text_retains_roles_and_reservation_is_not_generation(self):
        self.write("data/v3/real/holdout.jsonl", [
            {"id": "long_used", "report": "Shared source"},
            {"id": "long_reserved", "report": "Reserved source"},
        ])
        self.write("data/v3/real/main20.jsonl", [{"id": "long_used", "report": "SHARED source"}])
        self.write("results/v3/real-data/collected/v3/outputs.jsonl", [
            {"id": "long_used", "report": "Shared source", "result": {"text": "A prediction"}}
        ])
        inv = Inventory(self.root)
        for p in ("data/v3/real/holdout.jsonl", "data/v3/real/main20.jsonl",
                  "results/v3/real-data/collected/v3/outputs.jsonl"):
            inv.jsonl(p)
        rows = {r["report_hash"]: r for r in inv.finish()["reports"]}
        used = rows[fingerprint("Shared source")]
        self.assertEqual(used["usage_status"], "actual_used")
        self.assertEqual(len(used["events"]), 3)
        self.assertFalse(used["review_evidence_present"])
        self.assertEqual(rows[fingerprint("Reserved source")]["usage_status"], "reserved_only")

    def test_teacher_output_does_not_imply_viewing_or_review(self):
        path = "data/pilot_v1/train.zh.jsonl"
        self.write(path, [{"id": "train01", "report": "Private source",
                           "teacher_generation": {"text": "Private target"}}])
        inv = Inventory(self.root)
        inv.jsonl(path)
        row = inv.finish()["reports"][0]
        self.assertTrue(row["local_teacher_output_present"])
        self.assertFalse(row["local_output_present"])
        self.assertFalse(row["review_evidence_present"])
        self.assertEqual(row["human_or_project_ai_viewing_status"], "not_inferred")

    def test_downloaded_input_copy_is_not_generation(self):
        path = "results/v3/real-data/downloaded-run/inputs/final20.jsonl"
        self.write(path, [{"id": "long_copy", "report": "Copied input"}])
        inv = Inventory(self.root)
        inv.jsonl(path)
        row = inv.finish()["reports"][0]
        self.assertFalse(row["local_output_present"])
        self.assertEqual(row["usage_status"], "unknown")
        self.assertEqual(row["events"][0]["evidence"], "input_copy_present")
        self.assertFalse(any(i["code"] == "output_evidence_missing" for i in inv.issues))

    def test_no_source_prediction_or_bad_line_in_public_json(self):
        secret_report = "PRIVATE_REPORT_8675309"
        secret_prediction = "PRIVATE_PREDICTION_1234567"
        path = "results/v2/real-data/dev_probe/outputs.jsonl"
        self.write(path, [{"id": "test01", "report": secret_report,
                           "result": {"text": secret_prediction}}])
        with (self.root / path).open("a") as stream:
            stream.write('{"report": "MALFORMED_PRIVATE_TEXT"\n')
        inv = Inventory(self.root)
        inv.jsonl(path)
        public = json.dumps(inv.finish())
        for text in (secret_report, secret_prediction, "MALFORMED_PRIVATE_TEXT"):
            self.assertNotIn(text, public)
        self.assertIn(fingerprint(secret_report), public)
        self.assertTrue(any(i["code"] == "invalid_jsonl_record" for i in inv.issues))
        self.assertFalse(inv.finish()["summary"]["inventory_complete"])

    def test_unknown_schema_remains_gap(self):
        path = "results/v3/real-data/new_probe/outputs.jsonl"
        self.write(path, [{"unrecognized_text_field": "private"}, ["private"]])
        inv = Inventory(self.root)
        inv.jsonl(path)
        self.assertEqual(len(inv.issues), 2)
        self.assertEqual(inv.finish()["reports"], [])

    def test_correct_record_count_with_duplicate_reports_is_incomplete(self):
        path = "data/v3/real/holdout.jsonl"
        self.write(path, [{"id": f"long_{i}", "report": f"candidate {i if i < 39 else 0}"}
                          for i in range(40)])
        inv = Inventory(self.root)
        inv.jsonl(path)
        result = inv.finish()
        codes = {issue["code"] for issue in result["issues"]}
        self.assertIn("canonical_duplicate_report_hash", codes)
        self.assertNotIn("canonical_count_mismatch", codes)
        self.assertFalse(result["summary"]["inventory_complete"])

    def test_p0_inputs_are_excluded(self):
        self.write("data/p0/learning_packet.jsonl", [{"id": "new", "report": "NEW_P0_TEXT"}])
        inv = Inventory(self.root)
        manifest = inv.build()
        self.assertNotIn("data/p0/learning_packet.jsonl", manifest["sources"])
        self.assertNotIn(fingerprint("NEW_P0_TEXT"), json.dumps(manifest))

    def test_existing_output_directory_is_never_overwritten(self):
        out = self.root / "existing"
        out.mkdir()
        marker = out / "keep"
        marker.write_text("unchanged")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--root", str(self.root), "--out", str(out)])
        self.assertEqual(marker.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
