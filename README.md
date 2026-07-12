# MASP

Anonymized code for our EMNLP submission: **Mentalized Adversarial Self-Play for proactive dialogue**.

Datasets, model weights, and checkpoints are not included. This repository contains no author-identifying information.

## Layout

```text
src/masp/    Models, dialogue environment, BDI state, rewards, evaluation
scripts/     Training, labeling, and evaluation entry points
prompts/     Prompt templates
configs/     Configuration examples
docs/        Reproduction notes
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
cp scripts/env.example.sh scripts/env.local.sh   # then edit local paths
```

Data goes under `DATA_ROOT/{p4g,esconv,empathetic_dialogues,craigslist_bargain}/{train,valid,test}.json`; set `MODEL_PATH` to a compatible causal LM.

## Reproduction

Run the pipeline in order; every script documents its full CLI via `--help`:

```bash
python scripts/extract_bdi_labels.py --help        # Phase 0a: BDI silver labels
python scripts/train_phase0_teacher.py --help      # Phase 0b: teacher
python scripts/train_phase0_mentalization.py --help
python scripts/train_phase1_warmup.py --help       # Phase 1: BC warm-start
python scripts/train_phase2_selfplay.py --help     # Phase 2: adversarial self-play
python scripts/evaluate_masp.py --help             # Evaluation
```

See `docs/reproduction.md` for the settings to record when reproducing reported results.

## License

MIT (see `LICENSE`).
