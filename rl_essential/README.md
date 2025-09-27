# Reinforcement-Learning-tool-for-RNA-folding
A research sandbox to explore reinforcement learning techniques for RNA secondary structure construction.

# Goals
- Provide a clean environment API modeling folding as a sequential decision process.
- Keep dependencies minimal (only `numpy` + standard library).
- Offer baseline agents (random policy, tabular Q) for sanity checks.
- Make it easy to swap in better energy models and richer state encodings.


### Install
```bash
python -m venv .venv && source .venv/bin/activate # or Windows equivalent
pip install -e .
pytest # optional: run tests
