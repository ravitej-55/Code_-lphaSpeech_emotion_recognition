"""
Usage:
 python src/predict.py --model models/ser_cnn.h5 --wav-file samples/actor_01_sample.wav
"""
import argparse
import numpy as np
import tensorflow as tf
import librosa
import os

def extract_mfcc_for_predict(file_path, sr=22050, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, delta, delta2])
    if feat.shape[1] < max_len:
        pad_width = max_len - feat.shape[1]
        feat = np.pad(feat, ((0,0),(0,pad_width)), mode='constant')
    else:
        feat = feat[:, :max_len]
    return feat.astype('float32')

def main(model_path, wav_file, features_npz):
    # load label names
    data = np.load(features_npz, allow_pickle=True)
    labels = list(data['labels'])
    model = tf.keras.models.load_model(model_path)
    feat = extract_mfcc_for_predict(wav_file)
    # normalize same as train (sample-wise)
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    inp = np.expand_dims(feat, axis=0)
    preds = model.predict(inp)
    idx = np.argmax(preds, axis=1)[0]
    proba = preds[0][idx]
    print(f"Predicted: {labels[idx]} (prob={proba:.3f})")
    print("All probabilities:")
    for i,l in enumerate(labels):
        print(f"  {l}: {preds[0][i]:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav-file", required=True)
    parser.add_argument("--features-npz", required=True, help="path to features .npz (to load label names)")
    args = parser.parse_args()
    main(args.model, args.wav_file, args.features_npz)
