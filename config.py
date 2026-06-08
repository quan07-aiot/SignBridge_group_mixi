# ============================================================
# config.py — CẤU HÌNH DÙNG CHUNG cho training.py & webcam.py
# Chỉ chỉnh sửa file này, KHÔNG chỉnh trong từng file riêng.
# ============================================================

# --- Đường dẫn dữ liệu (chỉ training.py dùng) ---
VIDEO_DIR  = r"C:\Users\Home\OneDrive\VSL\Dataset\Video"
LABEL_FILE = r"C:\Users\Home\OneDrive\VSL\Dataset\Label\Label.csv"

# --- Đường dẫn file output (cả 2 file cùng dùng) ---
MODEL_PATH = "gesture_mediapipe_model.keras"
NAMES_PATH = "class_names.json"

# --- Thông số trích xuất (cả 2 file PHẢI dùng giống nhau) ---
MAX_FRAMES = 20        # số frame lấy mẫu từ mỗi video / mỗi cử chỉ webcam

# --- Thông số MediaPipe (cả 2 file PHẢI dùng giống nhau) ---
MAX_NUM_HANDS          = 2
MIN_DETECTION_CONF     = 0.5
MIN_TRACKING_CONF      = 0.5

# --- Thông số huấn luyện (chỉ training.py dùng) ---
EPOCHS      = 150
BATCH_SIZE  = 16
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

# --- Danh sách nhãn (chỉ training.py dùng để train) ---
# webcam.py tự đọc từ class_names.json sau khi train xong
ACTIONS = ['Chào', 'Tạm biệt', 'Cảm ơn']

# --- Thông số nhận diện webcam (chỉ webcam.py dùng) ---
VOTE_WINDOW          = 5     # số lần predict liên tiếp để voting
VOTE_MIN_AGREE       = 3     # số phiếu tối thiểu để chấp nhận kết quả
CONFIDENCE_THRESHOLD = 0.25  # ngưỡng softmax tối thiểu (0.0 – 1.0)
NO_HAND_RESET_LIMIT  = 15    # frames không thấy tay → reset buffer

# --- Giao diện ---
FONT_PATH    = r"C:\Windows\Fonts\arial.ttf"
FONT_SIZE    = 45
CAM_WIDTH    = 1280
CAM_HEIGHT   = 720
