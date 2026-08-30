#!/usr/bin/env python3
"""Standalone deterministic scorer for LDU-Bench Tasks A-D.

This module intentionally uses only the Python standard library. It reads
reference JSONL files and participant prediction JSONL files, then writes one
summary JSON file and one per-sample JSONL file for each selected task.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SUPPORTED_TASKS = ("A", "B", "C", "D")

REFERENCE_FILES = {
    "A": "task_A_binary_anomaly.jsonl",
    "B": "task_B_morphology_recognition.jsonl",
    "C": "task_C_coarse_localization.jsonl",
    "D": "task_D_cause_analysis.jsonl",
}

PREDICTION_FILES = {
    "A": "task_A_predictions.jsonl",
    "B": "task_B_predictions.jsonl",
    "C": "task_C_predictions.jsonl",
    "D": "task_D_predictions.jsonl",
}

# Frozen counts reported in the paper. The check is opt-in so that the scorer
# can also be exercised with small examples and development subsets.
RELEASE_COUNTS = {"A": 1761, "B": 1761, "C": 984, "D": 532}

TEXT_NORMALIZATION_RE = re.compile(
    r"[\s\-_，。、“”‘’'\";；:：,.!?！？/\(\)\[\]{}<>]+"
)


class ScoringError(ValueError):
    """Raised when a scorer input violates the public format."""


def safe_div(numerator: float, denominator: float) -> float:
    """Return zero for an empty denominator."""

    return 0.0 if denominator == 0 else numerator / denominator


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a UTF-8 JSONL file and attach concise, portable error messages."""

    if not path.is_file():
        raise ScoringError(f"missing input file: {path.name}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringError(
                    f"{path.name}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ScoringError(
                    f"{path.name}:{line_number}: each line must be a JSON object"
                )
            rows.append(row)
    return rows


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object."""

    if not path.is_file():
        raise ScoringError(f"missing input file: {path.name}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ScoringError(f"{path.name}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ScoringError(f"{path.name}: top-level JSON value must be an object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic, standards-compliant UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write deterministic UTF-8 JSONL, preserving input row order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            )
            handle.write("\n")


def _nonempty_string(row: dict[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ScoringError(f"{context}: {key} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


def _bbox_values(value: Any, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ScoringError(f"{context}: bounding box must be a four-item JSON array")
    try:
        bbox = [_finite_number(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScoringError(f"{context}: bounding-box coordinates must be finite numbers") from exc
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ScoringError(f"{context}: bounding box must have positive area")
    return bbox


def validate_reference_rows(task: str, rows: list[dict[str, Any]]) -> None:
    """Validate scorer-facing reference fields before any scores are written."""

    if not rows:
        raise ScoringError(f"{REFERENCE_FILES[task]}: reference file is empty")

    seen_task_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        context = f"{REFERENCE_FILES[task]}:{row_number}"
        task_id = _nonempty_string(row, "task_id", context)
        _nonempty_string(row, "sample_id", context)
        if task_id in seen_task_ids:
            raise ScoringError(f"{context}: duplicate reference task_id {task_id!r}")
        seen_task_ids.add(task_id)

        declared_task = row.get("task")
        if declared_task is not None and str(declared_task).strip().upper() != task:
            raise ScoringError(f"{context}: task field must be {task!r}")

        if task in {"A", "B"}:
            choices = row.get("choices")
            if (
                not isinstance(choices, list)
                or not choices
                or any(not isinstance(choice, str) or not choice for choice in choices)
            ):
                raise ScoringError(f"{context}: choices must be a non-empty string array")
            if len(set(choices)) != len(choices):
                raise ScoringError(f"{context}: choices must not contain duplicates")
            answer = _nonempty_string(row, "answer", context)
            if answer not in choices:
                raise ScoringError(f"{context}: answer must be present in choices")

        elif task == "C":
            image_size = row.get("image_size")
            if not isinstance(image_size, list) or len(image_size) != 2:
                raise ScoringError(f"{context}: image_size must be [width, height]")
            try:
                width, height = [_finite_number(item) for item in image_size]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ScoringError(
                    f"{context}: image_size values must be finite numbers"
                ) from exc
            if width <= 0 or height <= 0:
                raise ScoringError(f"{context}: image dimensions must be positive")
            bbox = _bbox_values(row.get("answer_bbox_xyxy"), context)
            if (
                bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] > width
                or bbox[3] > height
            ):
                raise ScoringError(f"{context}: reference bounding box exceeds image bounds")

        elif task == "D":
            _nonempty_string(row, "reference_cause_label", context)
            _nonempty_string(row, "reference_cause_analysis", context)


def index_predictions(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Index predictions; a repeated task_id deterministically uses its last row."""

    indexed: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    missing_task_id_count = 0
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            missing_task_id_count += 1
            continue
        if task_id in indexed:
            duplicate_count += 1
        indexed[task_id] = row
    return indexed, duplicate_count, missing_task_id_count


def normalize_label_prediction(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    value = row.get("prediction")
    return str(value).strip() if value is not None else ""


def score_classification_task(
    task: str,
    reference_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions, duplicate_count, missing_task_id_count = index_predictions(
        prediction_rows
    )
    valid_labels = sorted(
        {choice for row in reference_rows for choice in row["choices"]}
    )
    label_support = Counter(row["answer"] for row in reference_rows)
    true_positives: Counter[str] = Counter()
    false_positives: Counter[str] = Counter()
    false_negatives: Counter[str] = Counter()
    correct_by_label: Counter[str] = Counter()
    per_sample: list[dict[str, Any]] = []
    total_correct = 0
    answered_samples = 0
    invalid_predictions = 0

    for row in reference_rows:
        prediction_row = predictions.get(row["task_id"])
        prediction = normalize_label_prediction(prediction_row)
        answered = prediction_row is not None
        answered_samples += int(answered)

        is_valid = prediction in valid_labels
        if answered and not is_valid:
            invalid_predictions += 1

        is_correct = is_valid and prediction == row["answer"]
        if is_correct:
            total_correct += 1
            true_positives[row["answer"]] += 1
            correct_by_label[row["answer"]] += 1
        else:
            false_negatives[row["answer"]] += 1
            if is_valid:
                false_positives[prediction] += 1

        per_sample.append(
            {
                "task_id": row["task_id"],
                "sample_id": row["sample_id"],
                "reference": row["answer"],
                "prediction": prediction,
                "answered": answered,
                "valid_prediction": is_valid,
                "correct": is_correct,
            }
        )

    f1_by_label: dict[str, float] = {}
    accuracy_by_label: dict[str, float] = {}
    for label, support in label_support.items():
        f1_by_label[label] = safe_div(
            2 * true_positives[label],
            2 * true_positives[label]
            + false_positives[label]
            + false_negatives[label],
        )
        accuracy_by_label[label] = safe_div(correct_by_label[label], support)

    summary = {
        "task": task,
        "task_name": str(reference_rows[0].get("task_name") or ""),
        "total_samples": len(reference_rows),
        "answered_samples": answered_samples,
        "missing_predictions": len(reference_rows) - answered_samples,
        "invalid_predictions": invalid_predictions,
        "duplicate_task_ids": duplicate_count,
        "prediction_rows_missing_task_id": missing_task_id_count,
        "accuracy": safe_div(total_correct, len(reference_rows)),
        "macro_f1": safe_div(sum(f1_by_label.values()), len(f1_by_label)),
        "label_support": dict(label_support),
        "per_label_accuracy": accuracy_by_label,
        "per_label_f1": f1_by_label,
    }
    return summary, per_sample


def normalize_bbox_prediction(
    row: dict[str, Any] | None, image_size: Sequence[Any]
) -> tuple[list[float] | None, str | None]:
    if row is None:
        return None, "missing_prediction"

    bbox = row.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, "invalid_bbox_format"
    try:
        x_min, y_min, x_max, y_max = [_finite_number(value) for value in bbox]
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_bbox_value"

    width, height = [float(value) for value in image_size]
    clipped = [
        min(max(x_min, 0.0), width),
        min(max(y_min, 0.0), height),
        min(max(x_max, 0.0), width),
        min(max(y_max, 0.0), height),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None, "non_positive_bbox_area"
    return clipped, None


def bbox_area(bbox_xyxy: Sequence[float]) -> float:
    return max(0.0, bbox_xyxy[2] - bbox_xyxy[0]) * max(
        0.0, bbox_xyxy[3] - bbox_xyxy[1]
    )


def intersection_area(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    x_min = max(lhs[0], rhs[0])
    y_min = max(lhs[1], rhs[1])
    x_max = min(lhs[2], rhs[2])
    y_max = min(lhs[3], rhs[3])
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    return (x_max - x_min) * (y_max - y_min)


def compute_bbox_metrics(
    pred_bbox: Sequence[float],
    reference_bbox: Sequence[float],
    image_size: Sequence[Any],
) -> dict[str, float]:
    pred_area = bbox_area(pred_bbox)
    reference_area = bbox_area(reference_bbox)
    overlap = intersection_area(pred_bbox, reference_bbox)
    union_area = pred_area + reference_area - overlap
    total_pixels = float(image_size[0]) * float(image_size[1])
    negative_pixels = max(0.0, total_pixels - reference_area)
    false_positive_pixels = max(0.0, pred_area - overlap)

    miou = safe_div(overlap, union_area)
    gt_coverage = safe_div(overlap, reference_area)
    dicu = safe_div(2 * miou * gt_coverage, miou + gt_coverage)
    false_positive_rate = safe_div(false_positive_pixels, negative_pixels)
    p_auroc = 0.5 * (1.0 + gt_coverage - false_positive_rate)
    return {
        "miou": miou,
        "gt_coverage": gt_coverage,
        "dicu": dicu,
        "p_auroc": min(max(p_auroc, 0.0), 1.0),
    }


def score_localization_task(
    reference_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions, duplicate_count, missing_task_id_count = index_predictions(
        prediction_rows
    )
    metric_totals: Counter[str] = Counter()
    per_sample: list[dict[str, Any]] = []
    answered_samples = 0
    invalid_predictions = 0
    full_gt_cover_count = 0

    for row in reference_rows:
        prediction_row = predictions.get(row["task_id"])
        pred_bbox, error = normalize_bbox_prediction(
            prediction_row, row["image_size"]
        )
        answered = prediction_row is not None
        answered_samples += int(answered)

        if pred_bbox is None:
            if answered:
                invalid_predictions += 1
            metrics = {
                "miou": 0.0,
                "gt_coverage": 0.0,
                "dicu": 0.0,
                "p_auroc": 0.0,
            }
        else:
            reference_bbox = [float(value) for value in row["answer_bbox_xyxy"]]
            metrics = compute_bbox_metrics(
                pred_bbox, reference_bbox, row["image_size"]
            )
            if metrics["gt_coverage"] >= 0.999999:
                full_gt_cover_count += 1

        metric_totals.update(metrics)
        per_sample.append(
            {
                "task_id": row["task_id"],
                "sample_id": row["sample_id"],
                "reference_bbox_xyxy": row["answer_bbox_xyxy"],
                "prediction_bbox_xyxy": pred_bbox,
                "answered": answered,
                "valid_prediction": pred_bbox is not None,
                "error": error,
                **metrics,
            }
        )

    total_samples = len(reference_rows)
    summary = {
        "task": "C",
        "task_name": str(reference_rows[0].get("task_name") or ""),
        "total_samples": total_samples,
        "answered_samples": answered_samples,
        "missing_predictions": total_samples - answered_samples,
        "invalid_predictions": invalid_predictions,
        "duplicate_task_ids": duplicate_count,
        "prediction_rows_missing_task_id": missing_task_id_count,
        "mean_miou": safe_div(metric_totals["miou"], total_samples),
        "mean_gt_coverage": safe_div(
            metric_totals["gt_coverage"], total_samples
        ),
        "mean_dicu": safe_div(metric_totals["dicu"], total_samples),
        "mean_p_auroc": safe_div(metric_totals["p_auroc"], total_samples),
        "full_gt_cover_rate": safe_div(full_gt_cover_count, total_samples),
    }
    return summary, per_sample


def normalize_cause_text(text: Any) -> str:
    value = str(text or "").strip().lower()
    return TEXT_NORMALIZATION_RE.sub("", value)


def validate_rubric(
    payload: dict[str, Any], filename: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ScoringError(f"{filename}: labels must be a non-empty array")

    rule_map: dict[str, dict[str, Any]] = {}
    alias_lookup: dict[str, str] = {}
    for index, rule in enumerate(labels, start=1):
        context = f"{filename}:labels[{index}]"
        if not isinstance(rule, dict):
            raise ScoringError(f"{context}: label rule must be an object")
        canonical_label = _nonempty_string(rule, "canonical_label", context)
        if canonical_label in rule_map:
            raise ScoringError(f"{context}: duplicate canonical_label")
        _nonempty_string(rule, "supergroup", context)

        aliases = rule.get("aliases", [])
        keywords = rule.get("keywords", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise ScoringError(f"{context}: aliases must be a string array")
        if not isinstance(keywords, list) or any(
            not isinstance(keyword, str) or not keyword.strip()
            for keyword in keywords
        ):
            raise ScoringError(f"{context}: keywords must be a string array")

        normalized_rule = dict(rule)
        normalized_rule["canonical_label"] = canonical_label
        normalized_rule["aliases"] = [alias.strip() for alias in aliases]
        normalized_rule["keywords"] = [keyword.strip() for keyword in keywords]
        rule_map[canonical_label] = normalized_rule

        for alias in [canonical_label, *normalized_rule["aliases"]]:
            normalized_alias = normalize_cause_text(alias)
            previous = alias_lookup.get(normalized_alias)
            if previous is not None and previous != canonical_label:
                raise ScoringError(
                    f"{context}: alias {alias!r} is shared by multiple labels"
                )
            alias_lookup[normalized_alias] = canonical_label

    return rule_map, alias_lookup


def canonicalize_cause_label(
    text: Any, alias_lookup: dict[str, str]
) -> str | None:
    normalized_text = normalize_cause_text(text)
    if not normalized_text:
        return None
    if normalized_text in alias_lookup:
        return alias_lookup[normalized_text]

    best_alias = ""
    best_label: str | None = None
    for alias, canonical_label in alias_lookup.items():
        if alias and alias in normalized_text and len(alias) > len(best_alias):
            best_alias = alias
            best_label = canonical_label
    return best_label


def extract_keyword_hits(text: Any, keywords: Sequence[str]) -> set[str]:
    normalized_text = normalize_cause_text(text)
    return {
        keyword
        for keyword in keywords
        if normalize_cause_text(keyword)
        and normalize_cause_text(keyword) in normalized_text
    }


def normalize_task_d_prediction(row: dict[str, Any] | None) -> dict[str, str]:
    if row is None:
        return {"cause_label": "", "cause_analysis": "", "confidence": ""}
    return {
        "cause_label": str(row.get("cause_label") or "").strip(),
        "cause_analysis": str(row.get("cause_analysis") or "").strip(),
        "confidence": str(row.get("confidence") or "").strip().lower(),
    }


def score_cause_task(
    reference_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    rubric: dict[str, Any],
    rubric_filename: str = "task_d_cause_rubric.json",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rule_map, alias_lookup = validate_rubric(rubric, rubric_filename)
    predictions, duplicate_count, missing_task_id_count = index_predictions(
        prediction_rows
    )
    metric_totals: Counter[str] = Counter()
    per_sample: list[dict[str, Any]] = []
    answered_samples = 0
    invalid_predictions = 0
    exact_label_matches = 0
    same_supergroup_matches = 0

    for row_number, row in enumerate(reference_rows, start=1):
        prediction_row = predictions.get(row["task_id"])
        prediction = normalize_task_d_prediction(prediction_row)
        answered = prediction_row is not None
        answered_samples += int(answered)
        is_valid = bool(prediction["cause_label"] or prediction["cause_analysis"])
        if answered and not is_valid:
            invalid_predictions += 1

        reference_label = str(row["reference_cause_label"]).strip()
        reference_analysis = str(row["reference_cause_analysis"]).strip()
        reference_canonical = canonicalize_cause_label(
            reference_label, alias_lookup
        )
        if reference_canonical is None:
            raise ScoringError(
                f"{REFERENCE_FILES['D']}:{row_number}: reference_cause_label "
                "is absent from the Task D rubric"
            )
        prediction_canonical = canonicalize_cause_label(
            prediction["cause_label"], alias_lookup
        )

        reference_rule = rule_map[reference_canonical]
        prediction_rule = rule_map.get(prediction_canonical or "", {})
        reference_supergroup = str(reference_rule["supergroup"])
        prediction_supergroup = str(prediction_rule.get("supergroup") or "")

        semantic_match = 0.0
        if prediction_canonical == reference_canonical:
            semantic_match = 1.0
            exact_label_matches += 1
        elif prediction_supergroup and prediction_supergroup == reference_supergroup:
            semantic_match = 0.5
            same_supergroup_matches += 1

        reference_keywords_all = sorted(reference_rule["keywords"])
        reference_text = f"{reference_label} {reference_analysis}".strip()
        reference_keywords = sorted(
            extract_keyword_hits(reference_text, reference_keywords_all)
        )
        if not reference_keywords:
            reference_keywords = reference_keywords_all

        prediction_text = (
            f"{prediction['cause_label']} {prediction['cause_analysis']}".strip()
        )
        predicted_keywords = extract_keyword_hits(
            prediction_text, reference_keywords_all
        )
        matched_keywords = sorted(predicted_keywords.intersection(reference_keywords))
        keyword_precision = safe_div(len(matched_keywords), len(predicted_keywords))
        keyword_recall = safe_div(len(matched_keywords), len(reference_keywords))
        keyword_f1 = safe_div(
            2 * keyword_precision * keyword_recall,
            keyword_precision + keyword_recall,
        )
        task_d_score = 0.7 * semantic_match + 0.3 * keyword_f1

        metric_totals.update(
            {
                "semantic_match": semantic_match,
                "cause_keyword_precision": keyword_precision,
                "cause_keyword_recall": keyword_recall,
                "cause_keyword_f1": keyword_f1,
                "task_d_score": task_d_score,
            }
        )
        per_sample.append(
            {
                "task_id": row["task_id"],
                "sample_id": row["sample_id"],
                "reference_cause_label": reference_label,
                "reference_cause_analysis": reference_analysis,
                "reference_status": row.get("reference_status"),
                "prediction_cause_label": prediction["cause_label"],
                "prediction_cause_analysis": prediction["cause_analysis"],
                "prediction_confidence": prediction["confidence"],
                "reference_canonical_label": reference_canonical,
                "prediction_canonical_label": prediction_canonical,
                "answered": answered,
                "valid_prediction": is_valid,
                "matched_keywords": matched_keywords,
                "predicted_keywords": sorted(predicted_keywords),
                "reference_keywords": reference_keywords,
                "semantic_match": semantic_match,
                "cause_keyword_precision": keyword_precision,
                "cause_keyword_recall": keyword_recall,
                "cause_keyword_f1": keyword_f1,
                "task_d_score": task_d_score,
            }
        )

    total_samples = len(reference_rows)
    summary = {
        "task": "D",
        "task_name": str(reference_rows[0].get("task_name") or ""),
        "total_samples": total_samples,
        "answered_samples": answered_samples,
        "missing_predictions": total_samples - answered_samples,
        "invalid_predictions": invalid_predictions,
        "duplicate_task_ids": duplicate_count,
        "prediction_rows_missing_task_id": missing_task_id_count,
        "mean_semantic_match": safe_div(
            metric_totals["semantic_match"], total_samples
        ),
        "mean_cause_keyword_precision": safe_div(
            metric_totals["cause_keyword_precision"], total_samples
        ),
        "mean_cause_keyword_recall": safe_div(
            metric_totals["cause_keyword_recall"], total_samples
        ),
        "mean_cause_keyword_f1": safe_div(
            metric_totals["cause_keyword_f1"], total_samples
        ),
        "mean_task_d_score": safe_div(
            metric_totals["task_d_score"], total_samples
        ),
        "exact_label_match_rate": safe_div(exact_label_matches, total_samples),
        "same_supergroup_rate": safe_div(same_supergroup_matches, total_samples),
    }
    return summary, per_sample


def _unique_tasks(tasks: Sequence[str]) -> list[str]:
    result: list[str] = []
    for task in tasks:
        normalized = task.upper()
        if normalized not in SUPPORTED_TASKS:
            raise ScoringError(f"unsupported task: {task}")
        if normalized not in result:
            result.append(normalized)
    if not result:
        raise ScoringError("at least one task must be selected")
    return result


def run_scoring(
    references_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    tasks: Sequence[str] = SUPPORTED_TASKS,
    task_d_rubric_path: Path | None = None,
    check_release_counts: bool = False,
) -> dict[str, Any]:
    """Score selected tasks and write their deterministic result files."""

    selected_tasks = _unique_tasks(tasks)
    rubric: dict[str, Any] | None = None
    rubric_filename = "task_d_cause_rubric.json"
    if "D" in selected_tasks:
        rubric_path = task_d_rubric_path or (
            references_dir.parent / "rubrics" / "task_d_cause_rubric.json"
        )
        rubric = load_json_object(rubric_path)
        rubric_filename = rubric_path.name

    results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for task in selected_tasks:
        reference_rows = load_jsonl(references_dir / REFERENCE_FILES[task])
        validate_reference_rows(task, reference_rows)
        if check_release_counts and len(reference_rows) != RELEASE_COUNTS[task]:
            raise ScoringError(
                f"Task {task} reference count is {len(reference_rows)}; "
                f"expected frozen release count {RELEASE_COUNTS[task]}"
            )
        prediction_rows = load_jsonl(predictions_dir / PREDICTION_FILES[task])

        if task in {"A", "B"}:
            results[task] = score_classification_task(
                task, reference_rows, prediction_rows
            )
        elif task == "C":
            results[task] = score_localization_task(
                reference_rows, prediction_rows
            )
        else:
            assert rubric is not None
            results[task] = score_cause_task(
                reference_rows,
                prediction_rows,
                rubric,
                rubric_filename=rubric_filename,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for task in selected_tasks:
        summary, per_sample = results[task]
        summaries[task] = summary
        write_jsonl(
            output_dir / f"task_{task}_per_sample_results.jsonl", per_sample
        )

    overall: dict[str, Any] = {
        "tasks": summaries,
        "task_order": selected_tasks,
    }
    if check_release_counts:
        overall["release_count_check"] = {
            "passed": True,
            "expected": {task: RELEASE_COUNTS[task] for task in selected_tasks},
        }
    write_json(output_dir / "summary.json", overall)
    return overall


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score LDU-Bench Tasks A-D without external dependencies."
    )
    parser.add_argument(
        "--references-dir",
        type=Path,
        required=True,
        help="Directory containing the four reference task JSONL files.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing participant prediction JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for summary.json and per-sample result files.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(SUPPORTED_TASKS),
        choices=list(SUPPORTED_TASKS),
        help="Tasks to score; default: A B C D.",
    )
    parser.add_argument(
        "--task-d-rubric",
        type=Path,
        help=(
            "Task D rubric JSON. By default, rubrics/task_d_cause_rubric.json "
            "beside the references directory is used."
        ),
    )
    parser.add_argument(
        "--check-release-counts",
        action="store_true",
        help="Require frozen counts A=1761, B=1761, C=984, and D=532.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_scoring(
            references_dir=args.references_dir,
            predictions_dir=args.predictions_dir,
            output_dir=args.output_dir,
            tasks=args.tasks,
            task_d_rubric_path=args.task_d_rubric,
            check_release_counts=args.check_release_counts,
        )
    except (ScoringError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
