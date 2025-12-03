"""
Usage:
 python src/extract_features.py --data-dir data/RAVDESS --out-file data/ravdess_mfcc.npz
"""
import os
import argparse
import librosa
import numpy as np
from tqdm import tqdm

# RAVDESS emotion map (if using RAVDESS)
RAVDESS_EMO = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def parse_emotion_from_filename(filename):
    # RAVDESS pattern: Actor-01-01-01-01-01-01-01.wav (fields separated by -)
    parts = filename.split('-')
    if len(parts) >= 3:
        emo_code = parts[2]
        return RAVDESS_EMO.get(emo_code)
    return None

def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    # trim silence
    y, _ = librosa.effects.trim(y)
    # compute MFCC (n_mfcc x t)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # add deltas
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, delta, delta2])  # shape: (n_mfcc*3, t)
    # pad or truncate to fixed length (time axis)
    if feat.shape[1] < max_len:
        pad_width = max_len - feat.shape[1]
        feat = np.pad(feat, ((0,0),(0,pad_width)), mode='constant')
    else:
        feat = feat[:, :max_len]
    return feat

def main(data_dir, out_file, n_mfcc=40, max_len=174):
    X = []
    y = []
    labels_set = set()
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue
            fpath = os.path.join(root, fname)
            # determine label
            label = parse_emotion_from_filename(fname)
            if label is None:
                # fallback to directory name (last folder)
                label = os.path.basename(root)
            labels_set.add(label)
            try:
                feat = extract_mfcc(fpath, n_mfcc=n_mfcc, max_len=max_len)
            except Exception as e:
                print(f"Failed to process {fpath}: {e}")
                continue
            X.append(feat)
            y.append(label)
    if len(X) == 0:
        raise RuntimeError("No audio files found. Check data_dir path.")
    X = np.array(X)  # (N, channels, time)
    y = np.array(y)
    # Save label mapping
    labels = sorted(list(labels_set))
    label_to_idx = {l:i for i,l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y], dtype=np.int32)
    np.savez_compressed(out_file, X=X, y=y_idx, labels=labels)
    print(f"Saved features to {out_file}. X shape: {X.shape}, labels: {labels}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to dataset root (folders with .wav)")
    parser.add_argument("--out-file", required=True, help="Output .npz file path")
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=174)
    args = parser.parse_args()
    main(args.data_dir, args.out_file, n_mfcc=args.n_mfcc, max_len=args.max_len)
