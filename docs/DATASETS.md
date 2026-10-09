# Dataset Preparation and Paper Splits

All relative paths below are resolved from the release root. Dataset archives
are not redistributed. Obtain each dataset from its official provider and
follow its license.

| Paper benchmark | Evaluation task key | Expected root | Physical split | Primary metric |
|---|---|---|---|---|
| WHU Building | `building` | `data/whu_building` | `test` | IoU |
| SpaceNet 6 SAR | `sar_building` | `data/spacenet6_sar_512` | `test` | SAR IoU |
| WHU-CD | `change` | `data/whu_cd_raw` | `test` | IoU |
| LEVIR-CD+ | `change` | `data/LEVIR-CD+` | `test` | IoU |
| UrbanSARFloods | `sar_change` | `data/UrbanSARFloods` | held-out test events | mIoU |
| LoveDA | `semantic` | `data/LoveDA` | `Val` | mIoU |
| OpenEarthMap-RGB | `sar_semantic` | `data/OpenEarthMap` | `val` | mIoU |
| OpenEarthMap-SAR | `sar_semantic` | `data/OpenEarthMap-SAR` | `test` | mIoU |
| Potsdam | `height` | `data/potsdam_height_512_tiles` | 14 held-out tiles | RMSE (m) |
| DFC23 Track 2 | `height` | `data/dfc23_track2_512` | `test` | delta-1 |
| RRSIS-D | `referring` | `data/rrsisd_512` | `test` | cIoU / gIoU |
| LaSeRS | `referring` | `data/lasers_512` | released test partition | cIoU / gIoU |
| DIOR-R | `detection` | `data/dior_r_512` | `test` | oriented mAP50 |

## Preparation utilities

The `tools/` directory contains only conversion and cache-generation programs
needed by evaluation:

- `prepare_spacenet6_sar.py`: aligned RGB/SAR images, SAR normalization, and
  the deterministic manifest.
- `prepare_potsdam_height.py`: aligned RGB/nDSM 512-pixel tiles and the fixed
  24/14 tile split encoded by the paper config.
- `prepare_dfc23_track2.py`: RGB/SAR height samples and manifest.
- `prepare_rrsisd.py` and `prepare_lasers.py`: image/mask/expression manifests.
- `prepare_rrsisd_text_embeddings.py`: frozen Qwen prompt cache for either
  RRSIS-D or LaSeRS; use `--source rrsisd` or `--source lasers`.
- `prepare_dior_detection.py`: class-coded DIOR-R canvases and oriented-object
  metadata.
- `prepare_urbansar_test_512.py`: official held-out event tiling.
- `prepare_whu_change_labels.py`: deterministic WHU-CD label materialization.

Every utility exposes its exact arguments through `--help`. Examples:

```bash
python tools/prepare_spacenet6_sar.py --help
python tools/prepare_potsdam_height.py --help
python tools/prepare_dior_detection.py --help
python tools/prepare_rrsisd_text_embeddings.py \
  --config configs/uniflow_rs_512_2step_eval.yaml \
  --source rrsisd \
  --qwen-model-path pretrained/qwen3-4b-fp8
```

Do not regenerate a split with a different seed or substitute a resized copy
made by another pipeline. The evaluator assumes the manifests, interpolation,
normalization statistics, target palettes, and prompt-cache hashes produced by
these utilities.

## Evaluation-time spatial and target processing

- Model resolution is 512 x 512.
- Binary masks use nearest-neighbor label resizing and a fixed threshold of
  0.5.
- LoveDA and OpenEarthMap use the canonical VAE-aware semantic palette and are
  decoded back to dataset-native class IDs before metric computation.
- OpenEarthMap-SAR images larger than the model resolution use overlapping
  sliding windows with stride 256 and overlap blending.
- Heights use a log1p target encoding over 0-200 m; reported values are decoded
  back to meters. Potsdam predictions below 0.2 m are set to ground.
- DIOR-R uses the supplied VAE-aware 20-class palette, component area 64,
  score threshold 0.05, and rotated NMS IoU 0.1.
- Referring prompt caches are validated against both prompt count and SHA-256
  stored in each manifest.
