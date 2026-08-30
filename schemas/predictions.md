# LDU-Bench prediction schema

Predictions are UTF-8 JSON Lines files: one JSON object per non-empty line.
Every row needs an opaque `task_id` copied exactly from its reference task row.

| Task | File | Required prediction fields |
| --- | --- | --- |
| A | `task_A_predictions.jsonl` | `{"task_id":"A::LDU_000001","prediction":"good"}` |
| B | `task_B_predictions.jsonl` | `{"task_id":"B::LDU_000001","prediction":"<one choice from the task row>"}` |
| C | `task_C_predictions.jsonl` | `{"task_id":"C::LDU_000001","bbox_xyxy":[20,30,80,90]}` |
| D | `task_D_predictions.jsonl` | `{"task_id":"D::LDU_000001","cause_label":"<rubric label or alias>","cause_analysis":"<short explanation>","confidence":"medium"}` |

For Task A, `prediction` is `good` or `defect`. For Task C, coordinates are
finite numbers in original-image pixel coordinates ordered as
`[x_min, y_min, x_max, y_max]`; the scorer clips them to the image boundary and
rejects boxes with non-positive area. For Task D, `confidence` is optional and
does not affect the score. Additional fields are ignored.

Missing predictions score zero. A present row with an invalid task-specific
value also scores zero and is counted as invalid. If a `task_id` is repeated,
the last row is used and the duplicate is reported in `summary.json`.

From the extracted benchmark directory, run all four tasks with Python 3.10 or
newer:

```text
python score.py --references-dir tasks --predictions-dir examples/dummy_predictions --output-dir scores --check-release-counts
```

The optional release-count check requires A=1,761, B=1,761, C=984, and D=532.
By default, the Task D rubric is read from `rubrics/task_d_cause_rubric.json`;
use `--task-d-rubric` to override that location.

Without the count flag, small examples or selected tasks can be scored with
`--tasks A C`. The scorer uses only the Python standard library and performs no
model loading, inference, API, or network calls.

Outputs are `summary.json` plus `task_A_per_sample_results.jsonl` through
`task_D_per_sample_results.jsonl` for the selected tasks. A/B report accuracy
and macro-F1. C reports mean IoU, ground-truth coverage, DICU, and approximate
pixel AUROC. D reports semantic match, cause-keyword F1, and
`0.7 * semantic_match + 0.3 * cause_keyword_f1`.
