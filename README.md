## README.md
```markdown
# rna-rl

NN + MCTS sandbox for RNA secondary structure construction with Turner-like energetics.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[torch]
pytest -q
```

## Quickstart
Random rollout:
```bash
python -m scripts.play_random --seq GGGAAACCC
```
Double Q-learning:
```bash
python -m scripts.train_doubleq --seq GCAUCUAG --episodes 2000
```
AlphaZero-style training with PUCT:
```bash
python -m scripts.train_az --seq GGGAAACCC --iters 50 --episodes_per_iter 8 --batch 32 --device cpu
```

## Encoders
- `SimpleEncoder` (pooled token features)
- `GraphEncoder` (pairing graph + helix stack features)

Switch by modifying `learners/az_trainer.py` to import `GraphEncoder`.

## Energy Model
- Stacking (nearest neighbor), Hairpins (length + tetraloops), Internal/Bulge loops (length & asymmetry), Multibranch loops (affine + branch penalty), optional coaxial stacking heuristic.

## Caveats
This is a research scaffold; constants are approximate and for experimentation only.