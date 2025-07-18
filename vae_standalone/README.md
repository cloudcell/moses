# Standalone VAE for Moses

This folder contains a self-contained version of the Moses VAE model, extracted from the Moses codebase. It is designed to run independently for training and inference.

## Structure
- `model.py`      : VAE model definition
- `trainer.py`    : Training and evaluation logic
- `config.py`     : Configuration for VAE
- `misc.py`       : Miscellaneous helpers
- `utils.py`      : Utilities required for VAE (extracted from Moses)
- `models_storage.py`: Model checkpointing/storage logic (if required)
- `main.py`       : Entry point for running training/evaluation
- `requirements.txt`: Minimal dependencies

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Train VAE:
```bash
python main.py --train
```

Generate samples:
```bash
python main.py --generate
```

## Notes
- This codebase is independent of the rest of Moses. If you find missing imports, copy the relevant code from Moses into this folder.
- For molecule datasets and metrics, you may need to adapt or copy additional files.
