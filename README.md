# VLA

Vision-Language-Action research.

## Repo layout

Code is tracked in git. The following are intentionally **gitignored** (too large for git) and should live on local disk / a separate store:

- model weights & checkpoints (`*.pt`, `*.pth`, `*.safetensors`, `checkpoints/`, …)
- datasets & arrays (`data/`, `*.npy`, `*.hdf5`, …)
- experiment logs & media (`wandb/`, `logs/`, `*.mp4`, …)

See `.gitignore` for the full list.
