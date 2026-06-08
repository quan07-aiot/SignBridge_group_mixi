import os
import json
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==========================================
# ĐỌC CẤU HÌNH CHUNG TỪ config.py
# ==========================================
from config import (
    VIDEO_DIR, LABEL_FILE, MODEL_PATH, NAMES_PATH,
    MAX_FRAMES, MAX_NUM_HANDS, MIN_DETECTION_CONF, MIN_TRACKING_CONF,
    EPOCHS, BATCH_SIZE, TEST_SPLIT, RANDOM_SEED, ACTIONS,
)

ACTIONS = np.array(ACTIONS)

# ==========================================
# PHẦN 1: TRÍCH XUẤT TỌA ĐỘ BẰNG MEDIAPIPE
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF,
)


def extract_keypoints(frame: np.ndarray) -> np.ndarray:
    """Xử lý 1 frame, trả về mảng 126 tọa độ (2 bàn tay × 21 điểm × xyz)."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    data = np.zeros(126, dtype=np.float32)
    if results.multi_hand_landmarks:
        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            if i >= 2:
                break
            hand_data = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                dtype=np.float32,
            ).flatten()
            data[i * 63 : (i + 1) * 63] = hand_data
    return data


def process_video(video_path: str, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """
    Đọc video, lấy mẫu đều max_frames khung hình rồi trích xuất tọa độ.
    Trả về mảng (max_frames, 126).
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total > 0:
        indices = np.linspace(0, total - 1, max_frames, dtype=int)
    else:
        indices = np.zeros(max_frames, dtype=int)

    frames_data: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames_data.append(extract_keypoints(frame))
        elif frames_data:
            frames_data.append(frames_data[-1])
        else:
            frames_data.append(np.zeros(126, dtype=np.float32))

    cap.release()

    while len(frames_data) < max_frames:
        frames_data.append(frames_data[-1] if frames_data else np.zeros(126, dtype=np.float32))

    return np.array(frames_data, dtype=np.float32)   # (max_frames, 126)


# ==========================================
# PHẦN 2: TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================
def load_labels(label_file: str) -> dict[str, str]:
    """Đọc file CSV, trả về dict {tên_video: nhãn}."""
    df = pd.read_csv(label_file, header=None, dtype=str)

    if df.iloc[0, 1].strip().upper() == "VIDEO":
        df = df.iloc[1:]

    label_dict: dict[str, str] = {}
    for _, row in df.iterrows():
        video_name = str(row[1]).strip()
        label      = str(row[2]).strip()
        label_dict[video_name] = label
    return label_dict


def build_dataset(
    video_dir: str,
    label_dict: dict[str, str],
    actions: np.ndarray,
    max_frames: int = MAX_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Duyệt qua thư mục video, trích xuất đặc trưng và tạo X, y."""
    X_list: list[np.ndarray] = []
    y_list: list[int]        = []

    action_set = set(actions)
    video_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    print(f"Tìm thấy {len(video_files)} file video trong thư mục.")

    for i, video_name in enumerate(video_files, 1):
        label = label_dict.get(video_name)
        if label not in action_set:
            continue

        label_idx  = int(np.where(actions == label)[0][0])
        video_path = os.path.join(video_dir, video_name)

        landmarks = process_video(video_path, max_frames)
        X_list.append(landmarks)
        y_list.append(label_idx)

        if i % 10 == 0:
            print(f"  ... Đã xử lý {i}/{len(video_files)} video")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int32)
    return X, y


# ==========================================
# PHẦN 3: XÂY DỰNG MÔ HÌNH
# ==========================================
def build_model(num_frames: int, num_features: int, num_classes: int) -> tf.keras.Model:
    """Xây dựng mạng LSTM hai tầng với BatchNorm và Dropout."""
    model = Sequential([
        LSTM(64, return_sequences=True, activation="tanh",
             input_shape=(num_frames, num_features)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(128, return_sequences=False, activation="tanh"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ==========================================
# PHẦN 4: HUẤN LUYỆN
# ==========================================
def train(X: np.ndarray, y: np.ndarray, actions: np.ndarray) -> tf.keras.Model:
    """Chia tập train/val, tính class weight, huấn luyện với callbacks."""
    num_classes = len(actions)

    min_samples_for_split = num_classes * 2
    use_validation = len(X) >= min_samples_for_split * 2

    if use_validation:
        test_size = max(TEST_SPLIT, num_classes / len(X))
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        print(f"\nTập huấn luyện : {len(X_train)} mẫu")
        print(f"Tập validation : {len(X_val)} mẫu")
    else:
        X_train, y_train = X, y
        X_val,   y_val   = None, None
        print(f"\nDữ liệu ít ({len(X)} mẫu / {num_classes} lớp) → train trên toàn bộ, không chia validation.")
        print("  Gợi ý: thêm ít nhất 6 video mỗi lớp để dùng được validation.\n")

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights_arr))

    effective_batch = min(BATCH_SIZE, max(len(X_train) // 4, 1))
    if effective_batch != BATCH_SIZE:
        print(f"  Batch size tự điều chỉnh: {BATCH_SIZE} → {effective_batch}")

    model = build_model(
        num_frames=MAX_FRAMES,
        num_features=126,
        num_classes=num_classes,
    )
    model.summary()

    if use_validation:
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True,
                          verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                              min_lr=1e-5, verbose=1),
            ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True,
                            verbose=1),
        ]
        fit_kwargs = dict(validation_data=(X_val, y_val))
    else:
        callbacks = [
            ModelCheckpoint(MODEL_PATH, monitor="accuracy", save_best_only=True,
                            verbose=1),
        ]
        fit_kwargs = {}

    print("\nBắt đầu huấn luyện...\n")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=effective_batch,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
        **fit_kwargs,
    )
    return model


# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    print("Đang đọc file nhãn...")
    label_dict = load_labels(LABEL_FILE)
    print(f"  -> {len(label_dict)} nhãn được tải.")

    print("\nĐang trích xuất tọa độ từ video...")
    X, y = build_dataset(VIDEO_DIR, label_dict, ACTIONS, MAX_FRAMES)

    if len(X) == 0:
        print("\nLỖI: Không trích xuất được dữ liệu nào. "
              "Kiểm tra lại thư mục Video và file Label.csv!")
    else:
        print(f"\n--- TỔNG KẾT DỮ LIỆU ---")
        print(f"X : {X.shape}  (số video, số frame, số tọa độ)")
        print(f"y : {y.shape}")
        for idx, action in enumerate(ACTIONS):
            print(f"  Lớp {idx} '{action}': {(y == idx).sum()} mẫu")

        model = train(X, y, ACTIONS)

        with open(NAMES_PATH, "w", encoding="utf-8") as f:
            json.dump(ACTIONS.tolist(), f, ensure_ascii=False, indent=2)

        print(f"\n✓ Mô hình đã lưu tại : {MODEL_PATH}")
        print(f"✓ Nhãn đã lưu tại    : {NAMES_PATH}")