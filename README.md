# MASP: Mentalized Adversarial Self-Play for Proactive Dialogue

Official implementation of **MASP (Mentalized Adversarial Self-Play)** for proactive dialogue, accepted at **EMNLP 2026 Main Conference**.

MASP is a multi-agent self-play framework for training proactive dialogue agents through mentalization and adversarial interaction.

## Repository Structure

```text
src/masp/    Models, dialogue environment, BDI state, rewards, and evaluation
scripts/     Training, labeling, and evaluation entry points
prompts/     Prompt templates
configs/     Configuration examples
docs/        Reproduction notes
```

## Setup

Install the dependencies and MASP package:

```bash
pip install -r requirements.txt
pip install -e .
```

Create a local environment configuration:

```bash
cp scripts/env.example.sh scripts/env.local.sh
```

Then edit `scripts/env.local.sh` to configure local data and model paths.

Expected data structure:

```text
DATA_ROOT/
├── p4g/
├── esconv/
├── empathetic_dialogues/
└── craigslist_bargain/
    ├── train.json
    ├── valid.json
    └── test.json
```

Set `MODEL_PATH` to a compatible causal language model.

> **Note:** Datasets, model weights, and checkpoints are not included in this repository.

## Reproduction

The MASP training pipeline consists of BDI supervision, mentalization training, behavioral cloning warm-up, and adversarial self-play.

Run the following stages in order:

```bash
# Phase 0a: Extract BDI silver labels
python scripts/extract_bdi_labels.py --help

# Phase 0b: Train BDI teacher
python scripts/train_phase0_teacher.py --help

# Mentalization training
python scripts/train_phase0_mentalization.py --help

# Phase 1: Behavioral cloning warm-start
python scripts/train_phase1_warmup.py --help

# Phase 2: Mentalized adversarial self-play
python scripts/train_phase2_selfplay.py --help

# Evaluation
python scripts/evaluate_masp.py --help
```

Each script provides its complete command-line interface via `--help`.

For detailed reproduction settings and experimental configurations, see [`docs/reproduction.md`](docs/reproduction.md).

## Citation

If you find MASP useful in your research, please consider citing our EMNLP 2026 paper.

Citation information will be updated upon publication of the proceedings.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
