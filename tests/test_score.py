import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import score  # noqa: E402


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.references = self.root / "benchmark" / "tasks"
        self.predictions = self.root / "predictions"
        self.output = self.root / "scores"
        self._write_fixture()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_fixture(self):
        write_jsonl(
            self.references / score.REFERENCE_FILES["A"],
            [
                {
                    "task_id": "A::public_0001",
                    "task": "A",
                    "task_name": "binary_anomaly",
                    "sample_id": "public_0001",
                    "choices": ["good", "defect"],
                    "answer": "defect",
                },
                {
                    "task_id": "A::public_0002",
                    "task": "A",
                    "task_name": "binary_anomaly",
                    "sample_id": "public_0002",
                    "choices": ["good", "defect"],
                    "answer": "good",
                },
            ],
        )
        write_jsonl(
            self.predictions / score.PREDICTION_FILES["A"],
            [
                {"task_id": "A::public_0001", "prediction": "defect"},
                {"task_id": "A::public_0002", "prediction": "good"},
            ],
        )

        write_jsonl(
            self.references / score.REFERENCE_FILES["B"],
            [
                {
                    "task_id": "B::public_0001",
                    "task": "B",
                    "task_name": "morphology_recognition",
                    "sample_id": "public_0001",
                    "choices": ["normal", "feature_x"],
                    "answer": "feature_x",
                },
                {
                    "task_id": "B::public_0002",
                    "task": "B",
                    "task_name": "morphology_recognition",
                    "sample_id": "public_0002",
                    "choices": ["normal", "feature_x"],
                    "answer": "normal",
                },
            ],
        )
        write_jsonl(
            self.predictions / score.PREDICTION_FILES["B"],
            [
                {"task_id": "B::public_0001", "prediction": "feature_x"},
                {"task_id": "B::public_0002", "prediction": "feature_x"},
            ],
        )

        write_jsonl(
            self.references / score.REFERENCE_FILES["C"],
            [
                {
                    "task_id": "C::public_0001",
                    "task": "C",
                    "task_name": "coarse_localization",
                    "sample_id": "public_0001",
                    "image_size": [100, 100],
                    "answer_bbox_xyxy": [20, 20, 60, 60],
                },
                {
                    "task_id": "C::public_0002",
                    "task": "C",
                    "task_name": "coarse_localization",
                    "sample_id": "public_0002",
                    "image_size": [100, 100],
                    "answer_bbox_xyxy": [10, 10, 30, 40],
                },
            ],
        )
        write_jsonl(
            self.predictions / score.PREDICTION_FILES["C"],
            [{"task_id": "C::public_0001", "bbox_xyxy": [20, 20, 60, 60]}],
        )

        write_jsonl(
            self.references / score.REFERENCE_FILES["D"],
            [
                {
                    "task_id": "D::public_0001",
                    "task": "D",
                    "task_name": "cause_analysis",
                    "sample_id": "public_0001",
                    "reference_cause_label": "particle_contamination",
                    "reference_cause_analysis": "particle residue",
                    "reference_status": "reviewed",
                },
                {
                    "task_id": "D::public_0002",
                    "task": "D",
                    "task_name": "cause_analysis",
                    "sample_id": "public_0002",
                    "reference_cause_label": "cleaning_residue",
                    "reference_cause_analysis": "cleaning residue",
                    "reference_status": "reviewed",
                },
            ],
        )
        write_jsonl(
            self.predictions / score.PREDICTION_FILES["D"],
            [
                {
                    "task_id": "D::public_0001",
                    "cause_label": "particle",
                    "cause_analysis": "particle residue",
                    "confidence": "medium",
                },
                {
                    "task_id": "D::public_0002",
                    "cause_label": "particle",
                    "cause_analysis": "residue",
                },
            ],
        )
        rubric = {
            "task": "D",
            "labels": [
                {
                    "canonical_label": "particle_contamination",
                    "aliases": ["particle"],
                    "supergroup": "contamination",
                    "keywords": ["particle", "residue"],
                },
                {
                    "canonical_label": "cleaning_residue",
                    "aliases": ["residue issue"],
                    "supergroup": "contamination",
                    "keywords": ["cleaning", "residue"],
                },
            ],
        }
        rubric_path = (
            self.references.parent / "rubrics" / "task_d_cause_rubric.json"
        )
        rubric_path.parent.mkdir(parents=True, exist_ok=True)
        rubric_path.write_text(json.dumps(rubric), encoding="utf-8")

    def test_cli_scores_all_tasks_and_writes_expected_outputs(self):
        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            return_code = score.main(
                [
                    "--references-dir",
                    str(self.references),
                    "--predictions-dir",
                    str(self.predictions),
                    "--output-dir",
                    str(self.output),
                ]
            )

        self.assertEqual(return_code, 0)
        printed = json.loads(standard_output.getvalue())
        saved = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(printed, saved)
        self.assertEqual(saved["task_order"], ["A", "B", "C", "D"])
        self.assertEqual(saved["tasks"]["A"]["macro_f1"], 1.0)
        self.assertEqual(saved["tasks"]["B"]["accuracy"], 0.5)
        self.assertAlmostEqual(saved["tasks"]["B"]["macro_f1"], 1.0 / 3.0)
        self.assertEqual(saved["tasks"]["C"]["mean_dicu"], 0.5)
        self.assertEqual(saved["tasks"]["C"]["missing_predictions"], 1)
        self.assertAlmostEqual(saved["tasks"]["D"]["mean_task_d_score"], 0.775)
        self.assertEqual(saved["tasks"]["D"]["exact_label_match_rate"], 0.5)
        self.assertEqual(saved["tasks"]["D"]["same_supergroup_rate"], 0.5)

        expected_files = {"summary.json"} | {
            f"task_{task}_per_sample_results.jsonl" for task in score.SUPPORTED_TASKS
        }
        self.assertEqual(
            {path.name for path in self.output.iterdir()}, expected_files
        )

    def test_bbox_clipping_and_nonfinite_rejection(self):
        clipped, error = score.normalize_bbox_prediction(
            {"bbox_xyxy": [-5, -10, 110, 120]}, [100, 100]
        )
        self.assertEqual(clipped, [0.0, 0.0, 100.0, 100.0])
        self.assertIsNone(error)

        invalid, error = score.normalize_bbox_prediction(
            {"bbox_xyxy": [0, 0, math.inf, 10]}, [100, 100]
        )
        self.assertIsNone(invalid)
        self.assertEqual(error, "invalid_bbox_value")

    def test_duplicate_prediction_uses_last_row_and_is_reported(self):
        references = score.load_jsonl(
            self.references / score.REFERENCE_FILES["A"]
        )
        predictions = [
            {"task_id": "A::public_0001", "prediction": "good"},
            {"task_id": "A::public_0001", "prediction": "defect"},
            {"prediction": "defect"},
        ]
        summary, per_sample = score.score_classification_task(
            "A", references, predictions
        )
        self.assertEqual(summary["duplicate_task_ids"], 1)
        self.assertEqual(summary["prediction_rows_missing_task_id"], 1)
        self.assertTrue(per_sample[0]["correct"])
        self.assertEqual(summary["missing_predictions"], 1)

    def test_release_count_guard_rejects_a_subset_with_wrong_size(self):
        with self.assertRaisesRegex(score.ScoringError, "expected frozen release count 1761"):
            score.run_scoring(
                references_dir=self.references,
                predictions_dir=self.predictions,
                output_dir=self.output,
                tasks=["A"],
                check_release_counts=True,
            )
        self.assertFalse(self.output.exists())

    def test_duplicate_reference_ids_are_rejected(self):
        path = self.references / score.REFERENCE_FILES["A"]
        rows = score.load_jsonl(path)
        rows[1]["task_id"] = rows[0]["task_id"]
        with self.assertRaisesRegex(score.ScoringError, "duplicate reference task_id"):
            score.validate_reference_rows("A", rows)


if __name__ == "__main__":
    unittest.main()
