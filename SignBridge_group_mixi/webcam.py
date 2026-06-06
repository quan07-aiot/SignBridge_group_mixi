import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import mediapipe as mp
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CẤU HÌNH CƠ BẢN
# ==========================================
MODEL_PATH = "gesture_mediapipe_model.h5"

# Cập nhật danh sách nhãn đúng theo thứ tự model vừa train
CLASS_NAMES = ['Địa chỉ', 'Không cho', 'Không nên', 'Mù chữ', 'Chào', 'Tạm biệt'] 

MAX_FRAMES = 20
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # Để vẽ bộ xương tay lên màn hình cho trực quan
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

print("Đang khởi động AI... Vui lòng đợi!")


try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("-> Đã tải xong não bộ AI!")
except Exception as e:
    print(f"LỖI CHI TIẾT TỪ HỆ THỐNG: {e}")
    print(f"Loại lỗi: {type(e).__name__}")
    exit()
def put_vietnamese_text(frame, text, position, font_path, font_size, color_rgb):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# MỞ WEBCAM VÀ NHẬN DIỆN
# ==========================================
cap = cv2.VideoCapture(0)
# Hàng đợi chứa các mảng 126 số thay vì chứa ảnh
frames_buffer = deque(maxlen=MAX_FRAMES)

print("\nĐã bật Camera. Nhấn phím 'q' để tắt.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # 1. Trích xuất tọa độ giống y hệt hàm trong training.py
    data = np.zeros(126)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Tùy chọn: Vẽ khung xương tay lên webcam để bạn nhìn thấy MediaPipe đang hoạt động
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if i > 1: break
            hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            data[i*63 : (i+1)*63] = hand_data
            
    frames_buffer.append(data)
    
    # 2. Bắt đầu dự đoán khi thu thập đủ 20 frames tọa độ
    if len(frames_buffer) == MAX_FRAMES:
        input_seq = np.array(frames_buffer)
        input_seq = np.expand_dims(input_seq, axis=0) # Mở rộng chiều (1, 20, 126) cho AI
        
        predictions = model.predict(input_seq, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx] * 100
        predicted_label = CLASS_NAMES[predicted_idx]
        
        if confidence > 70.0:
            color_rgb = (0, 255, 0)
            text_to_show = f"{predicted_label} ({confidence:.1f}%)"
        else:
            color_rgb = (255, 255, 0)
            text_to_show = "Đang chờ cử chỉ..."
            
        frame = put_vietnamese_text(frame, text_to_show, (10, 40), FONT_PATH, 40, color_rgb)
        
    cv2.imshow("Nhan Dien Ngon Ngu Cu Chi - VSL", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()