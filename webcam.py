import json
import sys
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# ĐỌC CẤU HÌNH CHUNG TỪ config.py
# ==========================================
from config import (
    MODEL_PATH, NAMES_PATH, FONT_PATH, FONT_SIZE,
    MAX_FRAMES, MAX_NUM_HANDS, MIN_DETECTION_CONF, MIN_TRACKING_CONF,
    VOTE_WINDOW, VOTE_MIN_AGREE, CONFIDENCE_THRESHOLD, NO_HAND_RESET_LIMIT,
    CAM_WIDTH, CAM_HEIGHT,
)

# ==========================================
# PHẦN 0: KHỞI TẠO TÀI NGUYÊN
# ==========================================
def load_class_names(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = json.load(f)
        print(f"✓ Nhãn đã tải: {names}")
        return names
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy '{path}'. Hãy chạy training.py trước!")
        sys.exit(1)


def load_model(path: str) -> tf.keras.Model:
    print("Đang tải mô hình AI...")
    try:
        model = tf.keras.models.load_model(path)
        print("✓ Tải mô hình thành công!\n")
        return model
    except Exception as e:
        print(f"LỖI tải mô hình: {e}")
        sys.exit(1)


CLASS_NAMES = load_class_names(NAMES_PATH)
model       = load_model(MODEL_PATH)

# MediaPipe — thông số lấy từ config, giống hệt training.py
mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF,
)

# ==========================================
# PHẦN 1: TRÍCH XUẤT TỌA ĐỘ
# Logic giống hệt training.py — KHÔNG được thay đổi độc lập
# ==========================================
def extract_keypoints(results) -> np.ndarray:
    """Trả về mảng 126 float32 từ kết quả MediaPipe đã xử lý sẵn."""
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


# ==========================================
# PHẦN 2: HÀM VẼ UI
# ==========================================
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
        except IOError:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def put_vietnamese_text(frame, text, position, font_size, color_rgb):
    """Vẽ chữ tiếng Việt lên frame (dùng Pillow để hỗ trợ Unicode)."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img_pil).text(position, text, font=_get_font(font_size), fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_confidence_bars(frame, predictions, class_names, x, y):
    """Vẽ thanh xác suất — dùng Pillow cho nhãn tiếng Việt hiển thị đúng."""
    bar_max_width = 200
    bar_height    = 22
    gap           = 8
    font_size_bar = 16
    best_idx      = int(np.argmax(predictions))

    # Vẽ nền panel mờ
    panel_h = len(class_names) * (bar_height + gap) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y - 8),
                  (x + bar_max_width + 200, y + panel_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    # Vẽ các thanh màu bằng OpenCV
    for i, prob in enumerate(predictions):
        bar_y     = y + i * (bar_height + gap)
        bar_width = int(prob * bar_max_width)
        color     = (0, 200, 80) if i == best_idx else (200, 120, 0)
        cv2.rectangle(frame, (x, bar_y), (x + bar_max_width, bar_y + bar_height), (60, 60, 60), -1)
        cv2.rectangle(frame, (x, bar_y), (x + bar_width,     bar_y + bar_height), color, -1)

    # Vẽ nhãn tiếng Việt bằng Pillow (convert 1 lần duy nhất)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = _get_font(font_size_bar)

    for i, (name, prob) in enumerate(zip(class_names, predictions)):
        bar_y     = y + i * (bar_height + gap)
        color_rgb = (100, 230, 100) if i == best_idx else (230, 230, 230)
        draw.text((x + bar_max_width + 8, bar_y + 3),
                  f"{name}: {prob*100:.1f}%", font=font, fill=color_rgb)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ==========================================
# PHẦN 3: VÒNG LẶP CHÍNH
# ==========================================
def run() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("LỖI: Không mở được camera!")
        return

    # --- Trạng thái ---
    frames_buffer: list[np.ndarray] = []
    vote_buffer = deque(maxlen=VOTE_WINDOW)

    current_label      = "Thực hiện hành động"
    current_confidence = 0.0
    current_color_bgr  = (200, 200, 0)

    no_hand_counter = 0
    is_collecting   = False
    last_probs      = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    print("\nĐã bật Camera.")
    print("  [SPACE] = Reset buffer  |  [q] = Thoát")
    print(f"  Ngưỡng tin cậy : {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Voting         : {VOTE_MIN_AGREE}/{VOTE_WINDOW} phiếu\n")

    def reset_state():
        nonlocal current_label, current_confidence, current_color_bgr, is_collecting
        frames_buffer.clear()
        vote_buffer.clear()
        current_label      = "Thực hiện hành động"
        current_confidence = 0.0
        current_color_bgr  = (200, 200, 0)
        is_collecting      = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Nền tối phía trên
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # --- MediaPipe ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands.process(img_rgb)
        img_rgb.flags.writeable = True

        hand_detected = bool(results.multi_hand_landmarks)

        if hand_detected:
            no_hand_counter = 0
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        else:
            no_hand_counter += 1

        # Reset nếu mất tay quá lâu
        if no_hand_counter >= NO_HAND_RESET_LIMIT:
            if frames_buffer:
                reset_state()
            no_hand_counter = 0

        # --- Tích lũy & predict ---
        if hand_detected:
            kp = extract_keypoints(results)
            frames_buffer.append(kp)
            is_collecting = True

            if len(frames_buffer) >= MAX_FRAMES:
                seq = np.expand_dims(
                    np.array(frames_buffer[:MAX_FRAMES], dtype=np.float32), axis=0
                )  # (1, MAX_FRAMES, 126)

                last_probs  = model.predict(seq, verbose=0)[0]
                pred_idx    = int(np.argmax(last_probs))
                confidence  = float(last_probs[pred_idx])

                vote_buffer.append(pred_idx)

                if len(vote_buffer) >= VOTE_MIN_AGREE:
                    counts = Counter(vote_buffer)
                    winner_idx, winner_votes = counts.most_common(1)[0]

                    if winner_votes >= VOTE_MIN_AGREE and confidence >= CONFIDENCE_THRESHOLD:
                        current_label      = CLASS_NAMES[winner_idx]
                        current_confidence = confidence
                        current_color_bgr  = (0, 230, 80)
                    elif confidence < CONFIDENCE_THRESHOLD:
                        current_label      = "Không rõ cử chỉ"
                        current_confidence = confidence
                        current_color_bgr  = (0, 120, 255)

                frames_buffer.clear()
                is_collecting = False

        # --- Vẽ UI ---
        # Thanh xác suất (góc phải) — tiếng Việt hiển thị đúng
        if np.any(last_probs > 0) and hand_detected:
            frame = draw_confidence_bars(frame, last_probs, CLASS_NAMES,
                                         x=w - 420, y=100)

        # Tiến trình thu thập (chỉ ASCII nên dùng cv2 được)
        collected     = len(frames_buffer)
        progress_text = f"Thu thap: {collected}/{MAX_FRAMES}" if is_collecting else "San sang..."
        cv2.putText(frame, progress_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        # Trạng thái tay
        num_hands = len(results.multi_hand_landmarks) if hand_detected else 0
        cv2.putText(frame, f"Tay: {num_hands}/2", (w - 120, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        # Kết quả chính (góc trên, chữ lớn, tiếng Việt đúng)
        if current_confidence > 0:
            display_text = f"{current_label}  ({current_confidence*100:.1f}%)"
        else:
            display_text = current_label

        # BGR → RGB cho Pillow
        pil_color = (current_color_bgr[2], current_color_bgr[1], current_color_bgr[0])
        frame = put_vietnamese_text(frame, display_text, (10, 15), FONT_SIZE, pil_color)

        cv2.imshow("Nhan Dien Ngon Ngu Cu Chi - VSL", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            reset_state()
            print("-> Đã reset buffer.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Đã tắt camera.")


if __name__ == "__main__":
    run()