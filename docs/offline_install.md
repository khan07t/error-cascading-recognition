# Installing on a server without internet

The plant servers had no internet access, so the application was shipped as a wheel together with every dependency and the BERT model files. These are the steps, in the order they were used.

## On a machine with internet

1. Build the package:
   ```
   python -m pip install build
   python -m build            # writes dist/central_logging-0.1.0-py3-none-any.whl
   ```
2. Download every dependency as a wheel, for the server's Python version and platform:
   ```
   pip download -d wheelhouse -r requirements.txt pip setuptools wheel
   ```
3. Put the BERT files (`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`) into the application folder under `scripts/hf_cache`, or point `ECR_BERT_MODEL` to a folder that holds them.
4. Copy `dist/`, `wheelhouse/` and the model files to the server.

## On the server

```
python -m venv env
env\Scripts\Activate.ps1                         # Windows; on Linux: source env/bin/activate
pip install --no-index --find-links=wheelhouse pip setuptools wheel
pip install --no-index --find-links=wheelhouse dist\central_logging-0.1.0-py3-none-any.whl
```

`--no-index` makes pip use only the local wheels. The text model loads BERT with `local_files_only`, so nothing is fetched at run time.

## Check the install

```
python -c "import central_logging; print(central_logging.__file__)"
central-logging --help
```

## Daily run

1. The SCADA export for each PLC goes into `Input_PLC` (plant-wide files for the multi-PLC module into `Input_Multi_PLC`).
2. `central-logging <application folder>`
3. Results land in `Output_PLC` and `Output_Multi_PLC`. `scripts/Lib/Logs/Logs.csv` shows every file with each stage's status and run time.
4. A file that failed is in `Error_Files`. Fix it, move it back to `Input_PLC` and run again; stages that already passed are skipped.

The trained text model (`scaler.pkl`, `autoencoder.pth`, `hdbscan_model.pkl`) has to be in `scripts/Lib/Processing_Files/TextEmbeddingModel` before the first run. Model 3's scaler, PCA and KMeans are created on the first run and updated by every run after it.

## Versions used on the plant servers

Python 3.8 with `numpy 1.24.3`, `pandas 2.0.3`, `scikit-learn 1.3.0`, `hdbscan 0.8.40`, `joblib 1.4.0`, `torch 2.4.1`, `transformers 4.44.2`. The repository is tested with current versions (Python 3.11).
