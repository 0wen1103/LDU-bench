# Data Statement

## Provenance and authorization

The source images are proprietary industrial inspection data. The data owner authorized their research use and the release of a de-identified benchmark for non-commercial research. The authorization record is retained by the authors and is not part of the public package.

## De-identification

The release uses newly assigned benchmark identifiers and file names. It excludes fab, equipment, batch, wafer, source-sample, recipe, process-log, equipment-state, local-path, account, endpoint, and runtime identifiers. The benchmark contains no personal data.

## Released content

The release package contains de-identified images, task manifests and labels, one pixel-level mask annotation for each Task C instance, the frozen Task D rubric, prediction schemas, deterministic scorers, and documentation. The package does not contain model-calling code, API configurations, experimental pipelines, internal review material, or raw operational metadata.

## Scope

Task D is an image-conditioned attribution benchmark. It does not provide process logs, recipe parameters, equipment states, or certified process-level root causes. Users must not treat Task D outputs as autonomous process decisions.

## Intended use

The benchmark is intended only for non-commercial scientific research, including evaluation of multimodal models, localization methods, terminology grounding, and image-conditioned structured prediction. `DATA_LICENSE.md` defines the binding use conditions.
