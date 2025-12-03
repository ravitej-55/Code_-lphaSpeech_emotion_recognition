"""
Usage:
 python src/train.py --features data/ravdess_mfcc.npz --model-path models/ser_cnn.h5
"""
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os

def build_cnn(input_shape, n_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # input: (channels, time)
        layers.Reshape((*input_shape, 1)),  # add channel
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(features_path, model_path, epochs=40, batch_size=32):
    data = np.load(features_path, allow_pickle=True)
    X = data['X']  # (N, channels, time)
    y = data['y']  # int labels
    labels = list(data['labels'])
    n_classes = len(labels)
    print("Labels:", labels)
    # swap axes to shape (N, channels, time) -> (N, channels, time)
    # Keras expects (H, W, C) later; we reshape in build
    X = X.astype('float32')
    # normalize per-sample
    X = (X - X.mean(axis=(1,2), keepdims=True)) / (X.std(axis=(1,2), keepdims=True) + 1e-6)
    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # one-hot
    enc = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    enc_val = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)
    input_shape = X_train.shape[1:]  # (channels, time)
    model = build_cnn(input_shape, n_classes)
    model.summary()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    cb = [
        callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, verbose=1)
    ]
    history = model.fit(X_train, enc, validation_data=(X_val, enc_val),
                        epochs=epochs, batch_size=batch_size, callbacks=cb)
    # save history
    np.savez(os.path.join(os.path.dirname(model_path), "training_history.npz"), history=history.history)
    print("Training finished. Model saved to", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to .npz features file")
    parser.add_argument("--model-path", default="models/ser_cnn.h5")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args.features, args.model_path, epochs=args.epochs, batch_size=args.batch_size)
