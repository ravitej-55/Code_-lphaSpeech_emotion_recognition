"""
Usage:
 python src/evaluate.py --features data/ravdess_mfcc.npz --model-path models/ser_cnn.h5 --output-dir reports
"""
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main(features_path, model_path, output_dir="reports"):
    data = np.load(features_path, allow_pickle=True)
    X = data['X'].astype('float32')
    y = data['y']
    labels = list(data['labels'])
    n_classes = len(labels)
    # normalize same as in train
    X = (X - X.mean(axis=(1,2), keepdims=True)) / (X.std(axis=(1,2), keepdims=True) + 1e-6)
    model = tf.keras.models.load_model(model_path)
    preds = model.predict(X)
    y_pred = np.argmax(preds, axis=1)
    print(classification_report(y, y_pred, target_names=labels))
    cm = confusion_matrix(y, y_pred)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    outpath = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(outpath, bbox_inches='tight')
    print("Saved confusion matrix to", outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    main(args.features, args.model_path, args.output_dir)
