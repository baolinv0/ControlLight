# Scene Semantics and Photometric Attribute Profiler

A multi-axis scene profiler for auditing the coverage of ControlLight-style training data.
The profiler classifies each **independent scene once** using the original input or the neutral
`a_000` image. It does not classify every brightness level independently, because brightness
editing would contaminate time, exposure, color, and scene labels.

## What it outputs

The profiler produces the following axes:

- Environment: `indoor`, `outdoor`, `transition`, `uncertain`
- Observable time: `day`, `twilight`, `night`, `unknown`, `uncertain`
- Content, multi-label: `portrait`, `city`, `nature`
- Display-domain dynamic-range difficulty: `low`, `medium`, `high`
- Apparent rendered illumination color: `warm`, `neutral`, `cool`, `mixed`, `unknown`
- Input brightness: `dark`, `medium`, `bright`
- Rendered exposure state: `underexposed`, `normal`, `overexposed`, `dual-risk`
- Elements: sky, light sources, and optionally reflective surfaces

The output deliberately distinguishes observable image properties from physical capture
properties:

- `dynamic_range_proxy` is **not** sensor dynamic range or scene radiance range.
- `apparent_illumination` is **not** physical correlated color temperature.
- Time is `unknown` for pure indoor scenes without visible external cues.
- Reflective surfaces are `not_evaluated` unless the open-vocabulary branch is enabled.

## Why the architecture is hybrid

No single model is reliable for all attributes:

1. **SigLIP 2** provides zero-shot semantic evidence for indoor/outdoor, time, content, and
   apparent color style.
2. **SegFormer-B5 ADE20K** provides fixed semantic regions such as sky, people, vegetation,
   roads, buildings, furniture, and lights.
3. **Photometric analysis** computes linear-light luminance, clipping, shadows, regional
   brightness span, and patch-level warm/cool clusters.
4. **Grounding DINO** optionally detects lights and reflective-surface concepts that are not
   reliably covered by ADE20K.
5. **SAM 2** optionally converts Grounding DINO boxes into masks.
6. A rule-fusion layer exposes confidence, evidence, conflicts, and review reasons instead of
   forcing every image into a confident label.

## Installation

The parent ControlLight environment already contains the required core packages. From the
repository root:

```bash
python -m pip install -e .
```

For standalone use:

```bash
python -m pip install -r tools/scene_profiler/requirements.txt
```

Model weights are downloaded from Hugging Face on first use.

## Quick start

Classify images directly:

```bash
python tools/scene_profiler/run_scene_profiler.py \
  --input /path/to/images \
  --output /path/to/scene_profile_results \
  --device cuda \
  --dtype bfloat16
```

Classify a ControlLight-style dataset root. The profiler automatically uses `a_000/`:

```bash
python tools/scene_profiler/run_scene_profiler.py \
  --input /path/to/dataset_root \
  --reference-level a_000 \
  --output /path/to/scene_profile_results \
  --device cuda \
  --dtype bfloat16
```

Debug on twenty images:

```bash
python tools/scene_profiler/run_scene_profiler.py \
  --input /path/to/dataset_root \
  --output /tmp/scene_profiler_debug \
  --max-images 20 \
  --save-overlays 20
```

Enable open-vocabulary reflection and light detection:

```bash
python tools/scene_profiler/run_scene_profiler.py \
  --input /path/to/dataset_root \
  --output /path/to/results \
  --enable-open-vocab \
  --device cuda
```

Enable SAM 2 mask refinement as well:

```bash
python tools/scene_profiler/run_scene_profiler.py \
  --input /path/to/dataset_root \
  --output /path/to/results \
  --enable-sam2 \
  --device cuda
```

`--enable-sam2` automatically enables Grounding DINO.

## Outputs

```text
scene_profile_results/
├── scene_profiles.jsonl
├── scene_profiles.csv
├── raw_features.jsonl
├── dataset_summary.json
├── errors.jsonl
├── overlays/                 # optional visual checks
└── masks/                    # optional compressed NPZ masks
```

### `scene_profiles.jsonl`

Full per-scene decisions, confidence, evidence, limitations, and review reasons.

### `scene_profiles.csv`

Compact table for dataset statistics and manual filtering.

### `raw_features.jsonl`

Raw semantic probabilities, segmentation ratios, photometric measurements, and optional
open-vocabulary detections. Keep this file so thresholds can be recalibrated without rerunning
all neural models.

### `dataset_summary.json`

Counts for all labels, multi-label content coverage, element coverage, review rate, and numeric
feature distributions.

## Calibration modes

The default configuration uses **fixed thresholds** so the resulting class distribution reflects
the dataset rather than being forced into predefined proportions.

For exploratory stratification only, dataset-relative quantiles can be enabled:

```bash
python tools/scene_profiler/run_scene_profiler.py \
  --input /path/to/dataset_root \
  --output /path/to/results_quantile \
  --dataset-relative-thresholds
```

Dataset-relative quantiles must not be used as evidence that a dataset has balanced absolute
brightness or dynamic-range coverage: quantile binning creates balanced bins by construction.
For publication-quality labels, calibrate fixed thresholds on a 200–300 image human-labeled gold
set and override `config/default.yaml` with a project-specific YAML.

## Recommended manual validation

Before relying on the aggregate statistics, review at least:

- low-confidence environment and time samples;
- all `transition` scenes;
- mixed-illumination samples;
- images close to brightness or DR thresholds;
- low SegFormer valid-pixel-ratio samples;
- reflection detections;
- a stratified sample from every output category.

Recommended validation metrics:

- Macro-F1 for indoor/outdoor and day/twilight/night;
- mAP for portrait/city/nature;
- weighted Cohen's kappa for ordered DR, brightness, and exposure labels;
- presence F1 and mask IoU for sky/light/reflection;
- expected calibration error for confidence values.

## Configuration

Copy and edit:

```text
tools/scene_profiler/config/default.yaml
```

Then run:

```bash
python tools/scene_profiler/run_scene_profiler.py \
  --input /path/to/data \
  --output /path/to/results \
  --config /path/to/custom.yaml
```

Only keys in the custom file need to be specified; it is deeply merged with the default config.

## Design boundaries and failure modes

1. **Rendered sRGB limitation**: physical scene radiance, sensor clipping, and real illuminant CCT
   cannot be recovered from an 8-bit rendered image.
2. **Domain shift**: zero-shot semantic scores must be checked on the target mobile-imaging data.
3. **Indoor time ambiguity**: a bright artificial-light interior is not evidence of daytime.
4. **Reflection ambiguity**: bright white objects and specular reflections are easily confused;
   open-vocabulary detections remain soft evidence.
5. **Segmentation sensitivity**: masks are used as regional evidence, not absolute truth. Low mask
   confidence triggers review.
6. **Threshold calibration**: defaults are engineering starting points, not universal standards.
7. **Multi-label content**: portrait, city, and nature can co-exist; they are never forced into a
   single softmax category.
8. **Dataset counting**: nine brightness levels from one input count as one independent scene.

## Tests

Core tests do not download model weights:

```bash
python -m unittest discover -s tools/scene_profiler/tests -v
```

## Default model checkpoints

- `google/siglip2-base-patch16-384`
- `nvidia/segformer-b5-finetuned-ade-640-640`
- Optional: `IDEA-Research/grounding-dino-tiny`
- Optional: `facebook/sam2.1-hiera-tiny`

## Re-label without rerunning neural models

After adjusting thresholds or fusion rules in a custom YAML, reuse the saved raw features:

```bash
python tools/scene_profiler/run_scene_profiler.py \
  --relabel-from /path/to/old_results/raw_features.jsonl \
  --output /path/to/recalibrated_results \
  --config /path/to/calibrated.yaml
```

This command does not load SigLIP2, SegFormer, Grounding DINO, or SAM 2.
