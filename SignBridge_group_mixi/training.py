import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# CẤU HÌNH CƠ BẢN
# ==========================================
VIDEO_DIR = r"C:\Users\Home\OneDrive\Nhận diện ngôn ngữ cử chỉ\Dataset\Video" 
LABEL_FILE = r"C:\Users\Home\OneDrive\Nhận diện ngôn ngữ cử chỉ\Dataset\Label\label.csv" 

MAX_FRAMES = 20      
# CHÚ Ý: Danh sách này PHẢI khớp 100% với CLASS_NAMES trong file webcam.py
ACTIONS = np.array(['Địa chỉ', 'Không cho', 'Không nên', 'Mù chữ', 'Chào', 'Tạm biệt'])

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# ==========================================
# PHẦN 1: HÀM TRÍCH XUẤT TỌA ĐỘ BẰNG MEDIAPIPE
# ==========================================
def extract_keypoints(frame):
    """Xử lý 1 frame ảnh, trả về mảng 126 tọa độ (2 bàn tay)"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Tạo mảng 126 số 0 (nếu không thấy tay, dữ liệu sẽ là số 0)
    data = np.zeros(126) 
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i > 1: break # Chỉ lấy tối đa 2 tay
            # Lấy 21 điểm x, y, z và làm phẳng thành mảng 1 chiều (63 số)
            hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            # Ghép vào mảng 126 số tổng
            data[i*63 : (i+1)*63] = hand_data
            
    return data

def process_video_to_landmarks(video_path, max_frames=20):
    """Đọc video và trích xuất tọa độ cho đủ 20 frames"""
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // max_frames, 1) if total_frames > 0 else 1
    
    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Thay vì lưu ảnh, ta trích xuất và lưu mảng 126 tọa độ
        landmarks = extract_keypoints(frame)
        frames_data.append(landmarks)
        
    cap.release()
    
    # Nếu video ngắn, không đủ max_frames, ta nhân bản frame cuối cùng
    while len(frames_data) < max_frames:
        if frames_data:
            frames_data.append(frames_data[-1])
        else:
            frames_data.append(np.zeros(126))
            
    return np.array(frames_data)

# ==========================================
# PHẦN 2: TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================
X = []
y = []
label_dict = {}

print("Đang đọc file nhãn label.csv...")
try:
    # Lấy nhãn từ file CSV (cột 1 là tên video, cột 2 là nhãn)
    df = pd.read_csv(LABEL_FILE, header=None)
    
    # Bỏ qua dòng tiêu đề nếu có
    if df.iloc[0, 1] == 'VIDEO':
        df = df.iloc[1:]
        
    for index, row in df.iterrows():
        video_name = str(row[1]).strip()
        label = str(row[2]).strip()
        label_dict[video_name] = label
        
    print(f"-> Đã trích xuất {len(label_dict)} nhãn từ file CSV.")
except Exception as e:
    print(f"LỖI khi đọc file CSV: {e}")
    exit()

print("\nĐang dùng MediaPipe quét qua các video... (Quá trình này sẽ mất một lúc)")
count_processed = 0

for video_name in os.listdir(VIDEO_DIR):
    if not video_name.endswith(('.mp4', '.avi', '.mov')):
        continue
        
    if video_name in label_dict:
        label = label_dict[video_name]
        
        # Chỉ lấy những video có nhãn nằm trong mảng ACTIONS
        if label in ACTIONS:
            label_idx = np.where(ACTIONS == label)[0][0]
            video_path = os.path.join(VIDEO_DIR, video_name)
            
            # Trích xuất dữ liệu: (20, 126)
            video_landmarks = process_video_to_landmarks(video_path, max_frames=MAX_FRAMES)
            
            X.append(video_landmarks)
            y.append(label_idx)
            
            count_processed += 1
            if count_processed % 10 == 0:
                print(f"... Đã xử lý {count_processed} video")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"\n--- TỔNG KẾT DỮ LIỆU ---")
print(f"Kích thước tập đầu vào X: {X.shape} (N video, 20 frames, 126 tọa độ)")
print(f"Kích thước tập đầu ra y: {y.shape}")

# ==========================================
# PHẦN 3 & 4: XÂY DỰNG AI & HUẤN LUYỆN
# ==========================================
if len(X) == 0:
    print("\nLỖI: Không trích xuất được dữ liệu nào. Hãy kiểm tra lại thư mục Video!")
else:
    # Mạng LSTM chuyên xử lý dữ liệu chuỗi (chuỗi 20 khung hình)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(MAX_FRAMES, 126)))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2)) # Chống học vẹt (overfitting)
    model.add(Dense(len(ACTIONS), activation='softmax')) # Lớp cuối có 6 nơ-ron

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("\nBắt đầu huấn luyện...")
    # Vì dữ liệu lúc này chỉ là các con số tọa độ nên train sẽ cực kỳ nhanh!
    model.fit(X, y, epochs=150, batch_size=4, validation_split=0.2)
    
    model.save("gesture_mediapipe_model.h5")
    print("\n-> Xong! Đã tạo file 'gesture_mediapipe_model.h5' chuẩn cho 6 từ.")