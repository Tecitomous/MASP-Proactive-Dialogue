# Reproduction notes

The public repository does not include datasets, model weights, checkpoints, or provider credentials. Place licensed inputs under the paths documented in the root README and set the corresponding environment variables in `scripts/env.local.sh`.

For a reproducible run, record:

- dataset version and split;
- base-model revision;
- Python and dependency versions;
- random seed;
- decoding temperature and maximum tokens;
- training phase and checkpoint directory;
- judge model and prompt revision.

Generated files should be written below `RUN_ROOT`. Keep them outside the source tree so that a clean checkout remains small and reviewable.
