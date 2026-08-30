# LDU-Bench

LDU-Bench is a diagnostic benchmark for multimodal model evaluation in lithography defect review. It evaluates four independently scored capabilities:

- **Task A:** binary defect triage;
- **Task B:** morphology recognition;
- **Task C:** coarse defect localization;
- **Task D:** image-conditioned cause attribution.

## Release

This repository contains the official LDU-Bench evaluation code and documentation. The de-identified benchmark archive is distributed from the [v1.0 GitHub Release](https://github.com/0wen1103/LDU-bench/releases/tag/v1.0).

The release contains only the de-identified benchmark, task specifications, prediction schemas, and deterministic evaluation code. It does not contain model-calling code, API configurations, experimental pipelines, internal review tools, or source-system identifiers.

## Benchmark package

The versioned dataset archive will contain:

| Component | Count | Description |
|---|---:|---|
| Task A | 1,761 | Binary defect-triage instances |
| Task B | 1,761 | Morphology-recognition instances on the same image set |
| Task C | 984 | Coarse-localization instances with pixel-level masks |
| Task D | 532 | Expert-reviewed image-conditioned cause references |

All public sample identifiers and file names are newly assigned benchmark IDs. Fab, equipment, batch, wafer, source-sample, local-path, recipe, process-log, and runtime identifiers are excluded.

## Evaluation

The public scorer accepts one JSONL prediction file per task and computes the deterministic metrics reported in the paper:

- macro-F1 for Tasks A and B;
- DICU for Task C;
- the frozen semantic-match and keyword-evidence rubric for Task D.

Prediction formats and a local smoke test are documented in [`schemas/predictions.md`](schemas/predictions.md). No commercial-model API or inference code is required by the scorer.

## Data access and licenses

The benchmark images, masks, annotations, and manifests are authorized for non-commercial research evaluation only and are governed by [`DATA_LICENSE.md`](DATA_LICENSE.md). Redistribution, commercial use, model training or fine-tuning, and attempts to identify the source are prohibited. The standalone evaluation code is released under the MIT License in [`LICENSE-CODE`](LICENSE-CODE).

Download `LDU-Bench-v1.0.zip` and its SHA-256 file from the [v1.0 release](https://github.com/0wen1103/LDU-bench/releases/tag/v1.0). The archive contains one pixel-level mask annotation for each of the 984 Task C instances.

See [`DATA_STATEMENT.md`](DATA_STATEMENT.md) for provenance, de-identification, scope, and use boundaries.

## Citation

Please cite the EMNLP 2026 Industry Track paper when using LDU-Bench. Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).
