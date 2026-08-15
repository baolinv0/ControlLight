# Reference-Guided Aligned Pseudo-GT Selector

This package selects one image from a nine-level, source-aligned brightness ladder. The generated reference is used only as a **region-level person-tone reference**. Selection never uses candidate/reference pixel-wise L1, L2, PSNR, or SSIM.

## Install

```bash
python -m pip install -e .
python -m pip install -e '.[test]'
```

## Input and masks

Each sample uses this layout:

```text
scene001/
├── source.png
├── reference.png
├── ladder/
│   ├── a_m100.png ... a_p100.png
└── provided_masks/
    ├── source_person.png
    ├── source_face.png       # optional
    ├── reference_person.png
    └── reference_face.png    # optional
```

Multiple detected people can be supplied as `source_person_0.png`, `source_person_1.png`, and similarly for the reference. A single exact mask takes precedence. The built-in file provider is deterministic and caches source masks, so one source segmentation is reused for all nine candidates. A production segmentation system can implement `MaskProvider` and inject it into `PseudoGTSelector`.

## Run one sample or a batch

```bash
python -m pseudo_gt_selector.run \
  --input-root /data/samples \
  --output-root /data/selection-output \
  --mask-dir-name provided_masks \
  --debug-overlay-limit 100 \
  --debug-overlay-seed 0
```

`--input-root` may point directly to one sample or to a directory of sample folders.

## Outputs

Valid selections produce a byte-for-byte copy of the chosen ladder file under `pseudo_gt/`, a complete JSON score record, a contact sheet with all nine raw score decompositions and ranks, and six region masks. Batch artifacts are written sample by sample. A bounded reservoir retains masks for only the reproducible random overlay sample (up to 100 by default); dataset summaries retain mask-free records. The limit and random seed are configurable. Dataset outputs include `selection_summary.csv` and `dataset_summary.json` with level histogram, score and margin percentiles, and region availability.

Samples with no source person, no reference person, ambiguous primary people, missing ladder levels, or non-aligned ladder dimensions receive their mandated invalid status in the score JSON and are not silently changed into global-brightness selection.

## Tests

```bash
pytest -q
python -m compileall pseudo_gt_selector tests
```

Synthetic tests cover five cases where global brightness favors the wrong candidate but person-centered tone matching selects the correct ladder level. Final accuracy targets still require the real-data human study described in the specification.

## External real-data acceptance gate

Code completion does not claim the data-dependent acceptance items. Final delivery still requires: 100 human-selected real samples; reported Top-1 and Top-1±1 agreement; 50–100 archived real visualizations; and representative cases for each specified failure category. These results must be measured from supplied real images and labels, not synthesized by the test suite.
