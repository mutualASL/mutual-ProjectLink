import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
import seaborn as sns
import matplotlib.pyplot as plt

# --- Custom Callback for Convergence ---
class EarlyStopOnConvergence(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy=0.97, consistency=3, monitor='val_accuracy'):
        super(EarlyStopOnConvergence, self).__init__()
        self.target_accuracy = target_accuracy
        self.consistency = consistency
        self.monitor = monitor
        self.consistent_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        val_acc = logs.get(self.monitor)
        if train_acc is None or val_acc is None: return
        if train_acc >= self.target_accuracy and val_acc >= self.target_accuracy:
            self.consistent_epochs += 1
            print(f"\nConvergence Condition Met: Epoch {self.consistent_epochs}/{self.consistency}")
        else:
            self.consistent_epochs = 0
        if self.consistent_epochs >= self.consistency:
            print(f"\nTraining stopped: Accuracy >= {self.target_accuracy*100}% for {self.consistency} epochs.")
            self.model.stop_training = True

# --- Configuration ---
IMG_SIZE = 200  # Reverted to 200x200 to match your data
DATA_DIR = 'roitraining_data_chest'
BATCH_SIZE = 64  # Increased to speed up training steps
EPOCHS = 100
NUM_CLASSES = 26
np.random.seed(42)
tf.random.set_seed(42)

# --- Data Loading ---
def load_data(data_dir):
    images, labels, class_names = [], [], sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print("Loading data...")
    for i, sign_name in enumerate(class_names):
        sign_dir = os.path.join(data_dir, sign_name)
        for filename in os.listdir(sign_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(sign_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Ensure 200x200
                    images.append(img)
                    labels.append(i)
    print(f"Total images loaded: {len(images)}")
    return np.array(images), np.array(labels), class_names

# --- Model Definition ---
def create_landmark_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),  # Reduced filters
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.35),
        Flatten(),
        Dense(64, activation='relu'),  # Reduced units
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = AdamW(learning_rate=0.0002)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Plot Confusion Matrix ---
def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved as '{save_path}'")

# --- Plot Normalized Confusion Matrix (percent per true class) ---
def plot_confusion_matrix_normalized(y_true, y_pred, class_names, save_path='confusion_matrix_normalized.png'):
    cm = confusion_matrix(y_true, y_pred).astype('float')
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100.0
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=100, cbar_kws={'label': '% of true class'})
    plt.title('Confusion Matrix (row-normalized %)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()
    print(f"Normalized confusion matrix saved as '{save_path}'")

# --- Plot Training Curves (accuracy + loss over epochs) ---
def plot_training_curves(history, save_path='training_curves.png'):
    epochs = range(1, len(history.history['accuracy']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(epochs, history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history.history['loss'], label='Train', linewidth=2)
    ax2.plot(epochs, history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved as '{save_path}'")

# --- Plot Per-Letter Accuracy Bar Chart ---
def plot_per_letter_accuracy(y_true, y_pred, class_names, save_path='per_letter_accuracy.png'):
    cm = confusion_matrix(y_true, y_pred)
    accs = []
    for i in range(len(class_names)):
        total = np.sum(cm[i, :])
        accs.append(cm[i, i] / total if total > 0 else 0)
    accs = np.array(accs)
    colors = ['#2e7d32' if a >= 0.95 else '#f9a825' if a >= 0.85 else '#c62828' for a in accs]

    plt.figure(figsize=(14, 5))
    bars = plt.bar(class_names, accs * 100, color=colors)
    plt.axhline(95, color='#888', linestyle='--', linewidth=1, label='95% line')
    for bar, a in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, a * 100 + 0.6,
                 f"{a*100:.0f}", ha='center', fontsize=8)
    plt.ylim(0, 105)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Letter')
    plt.title('Per-Letter Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Per-letter accuracy chart saved as '{save_path}'")

# --- Per-Letter Accuracy ---
def print_per_letter_accuracy(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    for i, class_name in enumerate(class_names):
        true_positives = cm[i, i]
        total_instances = np.sum(cm[i, :])
        accuracy = true_positives / total_instances if total_instances > 0 else 0
        print(f"Accuracy for {class_name}: {accuracy:.2f} ({true_positives}/{total_instances})")

# --- Main Execution ---
if __name__ == "__main__":
    X, y, class_names = load_data(DATA_DIR)

    if len(X) == 0:
        print("CRITICAL ERROR: No data found.")
    else:
        X_processed = (X.astype('float32') / 255.0).reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)
        X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
        y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)
        print(f"\nData split -> Training: {len(X_train)}, Validation: {len(X_val)}")

        # Compute class weights based on frequency
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / np.max(class_weights)  # Normalize to avoid extreme weights
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Minimal augmentation
        datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.02,
            height_shift_range=0.02,
            zoom_range=0.02
        )

        model = create_landmark_model((IMG_SIZE, IMG_SIZE, 1), NUM_CLASSES)
        model.summary()

        callbacks = [
            ModelCheckpoint('best_landmark_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
            EarlyStopOnConvergence(target_accuracy=0.97, consistency=3)
        ]

        print("\n--- Starting Training with Optimized 200x200 Strategy ---")
        history = model.fit(
            datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(X_val, y_val_cat),
            class_weight=class_weight_dict,
            callbacks=callbacks
        )

        print("\n--- Evaluating Best Model ---")
        model.load_weights('best_landmark_model.keras')
        val_loss, val_accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
        print(f"\nFinal Validation Accuracy: {val_accuracy * 100:.2f}%")

        print("\n--- Classification Report ---")
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val_cat, axis=1)
        print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

        print("\n--- Per-Letter Accuracy ---")
        print_per_letter_accuracy(y_true_classes, y_pred_classes, class_names)

        print("\n--- Generating Graphics ---")
        plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
        plot_confusion_matrix_normalized(y_true_classes, y_pred_classes, class_names)
        plot_training_curves(history)
        plot_per_letter_accuracy(y_true_classes, y_pred_classes, class_names)

        model.save('v3piroiasl_landmark_model.keras')
        print("\nModel saved as v3piroiasl_landmark_model.keras")
        with open('class_names.txt', 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        print("Class names saved to 'class_names.txt'")
