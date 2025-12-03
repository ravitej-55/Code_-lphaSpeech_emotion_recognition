# Speech Emotion Recognition (SER)

Recognize human emotions (happy, angry, sad, neutral, ...) from speech audio using MFCC features + a small CNN.

## Contents
- `src/extract_features.py` — extract MFCCs from WAV files into a .npz dataset file
- `src/train.py` — trains a Keras CNN and saves the model
- `src/evaluate.py` — evaluate model and produce confusion matrix
- `src/predict.py` — predict emotion for a single WAV file
- `notebooks/` — optional notebook for EDA and experiments

## Dataset
Download one of the public datasets (RAVDESS, TESS, EMO-DB). Put the extracted audio files under `data/`:

RAVDESS file naming convention is supported by the extractor (it parses emotion id if present).

## Quick setup
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
   FEARURE EXTRACTION

python src/extract_features.py --data-dir data/RAVDESS --out-file data/ravdess_mfcc.npz

TRAIN
python src/train.py --features data/ravdess_mfcc.npz --model-path models/ser_cnn.h5 --epochs 40 --batch-size 32

EVALUATE
python src/evaluate.py --features data/ravdess_mfcc.npz --model-path models/ser_cnn.h5 --output reports

    PREDICT SINGLE AUDIO
python src/predict.py --model models/ser_cnn.h5 --wav-file path/to/sample.wav
