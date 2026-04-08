import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.neural_network import MLPClassifier

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

RANDOM_STATE = 42
LABEL_NOISE = 0.12  # stronger than before to reduce overconfidence

# Use only tougher features
FEATURE_COLS = [
    "proto_id",
    "packet_size_mean",
    "packet_size_std",
    "inter_arrival_mean",
    "inter_arrival_std"
]

def add_label_noise(y, noise_fraction=0.08, random_state=42):
    y_noisy = y.copy()
    n = int(len(y_noisy) * noise_fraction)
    if n <= 0:
        return y_noisy

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(y_noisy), size=n, replace=False)
    y_noisy.iloc[idx] = 1 - y_noisy.iloc[idx]
    return y_noisy

def chrono_split(df, frac=0.7):
    if "flow_start_time" in df.columns:
        df = df.sort_values("flow_start_time").reset_index(drop=True)
    split_idx = int(len(df) * frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load base datasets
attack_df = pd.read_csv("datasets/attack_flows.csv")
normal_df = pd.read_csv("datasets/normal_flows.csv")
hard_normal_df = pd.read_csv("datasets/hard_normal_flows2.csv")

# Labels
attack_df["label"] = 1
normal_df["label"] = 0
hard_normal_df["label"] = 0

# Combine normal + hard normal for tougher learning
normal_all = pd.concat([normal_df, hard_normal_df], ignore_index=True)

# Clean
for df in [attack_df, normal_all]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURE_COLS, inplace=True)

# Chronological splits
normal_train, normal_test = chrono_split(normal_all, frac=0.7)
attack_train_full, attack_test_full = chrono_split(attack_df, frac=0.7)

# Balance train and test
n_train = min(len(normal_train), len(attack_train_full))
n_test = min(len(normal_test), len(attack_test_full))

attack_train = attack_train_full.sample(n=n_train, random_state=RANDOM_STATE)
attack_test = attack_test_full.sample(n=n_test, random_state=RANDOM_STATE)

normal_train = normal_train.sample(n=n_train, random_state=RANDOM_STATE)
normal_test = normal_test.sample(n=n_test, random_state=RANDOM_STATE)

train_df = pd.concat([normal_train, attack_train], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
test_df = pd.concat([normal_test, attack_test], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

X_train = train_df[FEATURE_COLS].copy()
y_train = train_df["label"].astype(int)

X_test = test_df[FEATURE_COLS].copy()
y_test = test_df["label"].astype(int)

# Robust scaling is better for flow outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train AE only on normal training traffic
X_train_normal = train_df[train_df["label"] == 0][FEATURE_COLS].copy()
X_train_normal_scaled = scaler.transform(X_train_normal)

input_dim = X_train_scaled.shape[1]
latent_dim = 2

input_layer = Input(shape=(input_dim,))
x = GaussianNoise(0.05)(input_layer)
x = Dense(8, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
latent = Dense(latent_dim, activation="relu", name="latent")(x)
x = Dense(8, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(latent)
output_layer = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

autoencoder.fit(
    X_train_normal_scaled,
    X_train_normal_scaled,
    epochs=60,
    batch_size=128,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Reconstruction errors
X_train_recon = autoencoder.predict(X_train_scaled, verbose=0)
X_test_recon = autoencoder.predict(X_test_scaled, verbose=0)

train_error = np.mean(np.square(X_train_scaled - X_train_recon), axis=1).reshape(-1, 1)
test_error = np.mean(np.square(X_test_scaled - X_test_recon), axis=1).reshape(-1, 1)

# Two-stage classifier input = reduced features + anomaly score
X_train_final = np.hstack([X_train_scaled, train_error])
X_test_final = np.hstack([X_test_scaled, test_error])

# Add stronger label noise to avoid overconfident fit
y_train_noisy = add_label_noise(y_train, noise_fraction=LABEL_NOISE, random_state=RANDOM_STATE)

clf = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.01,
    batch_size=128,
    learning_rate="adaptive",
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=12,
    random_state=RANDOM_STATE
)

clf.fit(X_train_final, y_train_noisy)

pred = clf.predict(X_test_final)
prob = clf.predict_proba(X_test_final)[:, 1]

# Save
autoencoder.save("models/autoencoder_model.h5")
joblib.dump(clf, "models/ae_mlp_classifier.pkl")
joblib.dump(scaler, "models/ae_scaler.pkl")
joblib.dump(FEATURE_COLS, "models/ae_feature_columns.pkl")

with open("results/ae_model_report.txt", "w") as f:
    f.write("=== AUTOENCODER + MLP CLASSIFIER (MAIN TWO-STAGE) ===\n")
    f.write(f"Features used: {FEATURE_COLS}\n")
    f.write(f"Train rows: {len(train_df)}\n")
    f.write(f"Test rows: {len(test_df)}\n")
    f.write(f"Accuracy: {accuracy_score(y_test, pred):.4f}\n")
    f.write(f"Precision: {precision_score(y_test, pred):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, pred):.4f}\n")
    f.write(f"F1: {f1_score(y_test, pred):.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, pred)))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, pred))

print("Autoencoder + MLP training completed.")
print("Saved: results/ae_model_report.txt")
