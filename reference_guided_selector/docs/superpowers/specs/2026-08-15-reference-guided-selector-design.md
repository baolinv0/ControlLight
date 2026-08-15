# Reference-Guided Aligned Pseudo-GT Selector Design

## Authority and scope

`project_sources/02-spec.txt` is the authoritative product specification. `project_sources/01-method.txt` is supporting design guidance. This document records the implementation decisions needed to turn them into a standalone Python package.

V1 selects one of the nine source-aligned brightness-ladder images by comparing person-centric tone descriptors with a non-aligned reference. It does not generate the ladder, register images, compare reference and candidates pixel-wise, learn a selector, use a VLM, or make color part of the selection score.

## Architecture

The package is split into configuration and data contracts, input discovery, mask extraction, local-background construction, photometric descriptors, scoring and selection, artifact rendering, dataset statistics, and a CLI. A `MaskProvider` protocol separates the selector from a concrete segmentation backend. The package includes a deterministic mask-file provider for production pipelines and tests; callers can inject their existing person/face segmentation implementation without changing selection logic.

Source masks are computed once and reused for all ladder levels. Reference masks are computed independently. Candidate descriptors are measured in linear-RGB luminance followed by log2 encoding. The score is built only from person, face, local-background, and person/background-relation descriptors, with unavailable components removed and remaining weights normalized.

## Outputs and invalid samples

For a valid sample, the selected pseudo-GT is copied byte-for-byte from the selected ladder path. JSON scores, masks, visualization, CSV summary, and dataset JSON statistics are written under the directory layout required by the specification.

No person, invalid reference person, ambiguous primary person, missing ladder levels, or source/ladder size mismatch produce explicit invalid statuses and score records. They do not produce a formal pseudo-GT. Missing face and invalid local background are supported fallbacks, not invalid samples.

## Engineering constraints

- Functional implementation has priority; only specification-mandated validation is added.
- No hashes or SHA256 are implemented.
- No speculative security checks or repeated defenses for practically impossible cases.
- Rubrics remain semantic and evidence-based rather than mechanically counting incidental implementation details.
- Tests use synthetic images and deterministic masks, covering the required five person-dominance cases, person/background relation, face fallback, byte-for-byte selection, invalid statuses, summaries, visualization, and CLI batch flow.

