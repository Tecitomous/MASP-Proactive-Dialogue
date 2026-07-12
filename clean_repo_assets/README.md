# MASP

Mentalized Adversarial Self-Play for proactive dialogue.

This repository is a clean research-code release of the MASP training and evaluation pipeline. It contains the model code, prompt templates, configuration examples, and reproducibility utilities. Datasets, model weights, experiment logs, checkpoints, and private service configuration are intentionally not included.

## Repository layout

```text
src/masp/       Core models, dialogue environment, BDI state, rewards, and evaluation
scripts/        Training, labeling, conversion, evaluation, and verification entry points
prompts/        BDI, rationality, role, and goal prompt templates
configs/        Public configuration examples
tests/          Lightweight import and interface checks
docs/           Reproduction notes and data preparation guidance
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp scripts/env.example.sh scripts/env.local.sh
```

Edit `scripts/env.local.sh` with local model and data paths. The local file is ignored by Git and must never contain credentials committed to the repository.

## Expected local inputs

The code expects the following paths to be supplied by the user:

```text
DATA_ROOT/
  p4g/{train,valid,test}.json
  esconv/{train,valid,test}.json
  empathetic_dialogues/{train,valid,test}.json
  craigslist_bargain/{train,valid,test}.json
MODEL_PATH/              A compatible causal language model
RUN_ROOT/                Writable output directory for caches and checkpoints
```

Dataset downloads and licenses are the responsibility of the user. Do not redistribute datasets or model weights with this repository.

## Reproduction entry points

Each entry point exposes its complete CLI through `--help`:

```bash
python scripts/extract_bdi_labels.py --help
python scripts/convert_src_bdi_to_masp_cache.py --help
python scripts/train_phase0_teacher.py --help
python scripts/train_phase0_mentalization.py --help
python scripts/train_phase1_warmup.py --help
python scripts/train_phase2_selfplay.py --help
python scripts/evaluate_masp.py --help
```

Use the same random seed, model revision, dataset split, and decoding parameters when reproducing reported results. Save generated artifacts below `RUN_ROOT` rather than inside the source tree.

## Verification

```bash
bash scripts/check_install.sh
python -m compileall -q src scripts
```

The checks do not download data or call an external model service.

## License

Code in this repository is released under the MIT License. Third-party datasets, base models, and optional dependencies remain subject to their own licenses.
