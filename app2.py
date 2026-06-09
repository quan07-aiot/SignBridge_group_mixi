import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import Counter, deque

# --- THƯ VIỆN TEXT-TO-SPEECH ---
import threading
from gtts import gTTS
import pygame
import io  
# -------------------------------

from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen

# Mô phỏng kích thước điện thoại dọc ngay trên PC khi chạy test
from kivy.core.window import Window
Window.size = (360, 640)

# Đọc cấu hình từ file config.py của bạn
import config

# Khởi tạo MediaPipe ẩn ngầm
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=config.MAX_NUM_HANDS,
    min_detection_confidence=config.MIN_DETECTION_CONF,
    min_tracking_confidence=config.MIN_TRACKING_CONF
)

# GIAO DIỆN SIGNBRIDGE VỚI TYPOGRAPHY ĐƯỢC CHUỐT ĐẸP
KV = f'''
<VSLAppScreen>:
    MDNavigationLayout:
        
        # ---------------- MÀN HÌNH CHÍNH ----------------
        MDScreenManager:
            MDScreen:
                md_bg_color: 0.95, 0.96, 0.98, 1  # Nền xám xanh nhạt chuẩn mẫu

                MDBoxLayout:
                    orientation: 'vertical'
                    padding: "16dp"
                    spacing: "16dp"

                    # Thanh Action Bar trên cùng thanh lịch
                    MDBoxLayout:
                        size_hint_y: None
                        height: "44dp"
                        orientation: 'horizontal'
                        spacing: "8dp"
                        
                        MDIconButton:
                            icon: "menu"
                            pos_hint: {{"center_y": .5}}
                            user_font_size: "22sp"
                            theme_text_color: "Custom"
                            text_color: 0.1, 0.2, 0.3, 1
                            on_release: nav_drawer.set_state("open")
                            
                        MDLabel:
                            text: "Signbridge Mobile"
                            font_style: "H6"
                            bold: True  # Làm đậm tên thương hiệu mới
                            theme_text_color: "Primary"
                            pos_hint: {{"center_y": .5}}

                    # 1. Khung Camera hình vuông 1:1
                    MDCard:
                        size_hint: 1, None
                        height: self.width  
                        elevation: 1
                        radius: [24, ]     
                        md_bg_color: 0, 0, 0, 1
                        overflow: 'hidden'
                        Image:
                            id: camera_feed
                            allow_stretch: True
                            keep_ratio: False  

                    # 2. Hộp kết quả dịch mỏng gọn (Chuốt lại font chữ phía trong)
                    MDCard:
                        size_hint_y: None
                        height: "85dp"  
                        elevation: 1
                        radius: [16, ]
                        padding: ["16dp", "8dp", "16dp", "8dp"]
                        md_bg_color: 1, 1, 1, 1
                        orientation: "horizontal"
                        pos_hint: {{"center_x": .5}}
                        
                        MDBoxLayout:
                            orientation: 'vertical'
                            size_hint_x: 0.38
                            spacing: "2dp"
                            pos_hint: {{"center_y": .5}}
                            
                            MDLabel:
                                text: "KẾT QUẢ DỊCH"
                                font_style: "Overline"
                                theme_text_color: "Secondary"
                                bold: True
                                font_size: "11sp"  # Thu nhỏ nhẹ chữ overline nhìn tinh tế hơn
                                
                            MDLabel:
                                id: status_tag
                                text: "Hệ thống sẵn sàng"
                                font_style: "Caption"
                                theme_text_color: "Hint"

                        # Vạch ngăn dọc mảnh
                        MDWidget:
                            size_hint_x: None
                            width: "1dp"
                            md_bg_color: 0.9, 0.92, 0.95, 1
                            size_hint_y: 0.6
                            pos_hint: {{"center_y": .5}}

                        # Chữ hiển thị kết quả chính (Đã căn chỉnh font to và bo tròn đẹp)
                        MDLabel:
                            id: result_label
                            text: "..."
                            halign: "center"
                            valign: "middle"
                            font_style: "H4"  # Font dạng tiêu đề lớn, nét dày và bo mịn cực đẹp
                            bold: True
                            theme_text_color: "Primary"
                            size_hint_x: 0.62

                    # 3. NÚT BẮT ĐẦU DỊCH DẠNG VIÊN THUỐC CAPSULE
                    MDBoxLayout:
                        orientation: 'vertical'
                        size_hint_y: 1  
                        padding: [0, 0, 0, "8dp"]
                        
                        MDWidget:
                            size_hint_y: 1  

                        MDRaisedButton:
                            id: translate_btn
                            text: "BẮT ĐẦU DỊCH"
                            size_hint_x: 1
                            height: "54dp"
                            radius: [27, ]  
                            font_size: "16sp"
                            bold: True  # Chữ trên nút in đậm hiện đại
                            elevation: 0    
                            md_bg_color: 0.12, 0.53, 0.68, 1  
                            on_release: root.toggle_translation()

        # ---------------- THANH SIDEBAR ẨN ----------------
        MDNavigationDrawer:
            id: nav_drawer
            radius: (0, 24, 24, 0) 
            md_bg_color: 0.96, 0.97, 0.99, 1 
            size_hint_x: 0.75  

            MDBoxLayout:
                orientation: 'vertical'
                padding: "16dp"
                spacing: "12dp"

                # Tiêu đề Sidebar Signbridge mới
                MDBoxLayout:
                    orientation: 'vertical'
                    size_hint_y: None
                    height: "90dp"
                    padding: ["12dp", "16dp", 0, "16dp"]
                    spacing: "4dp"
                    
                    MDLabel:
                        text: "SIGNBRIDGE"
                        font_style: "Button"
                        bold: True
                        font_size: "19sp"  
                        theme_text_color: "Primary"
                        
                    MDLabel:
                        text: "Hỗ trợ kết nối ngôn ngữ"
                        font_style: "Caption"
                        theme_text_color: "Secondary"

                MDWidget:
                    size_hint_y: None
                    height: "1dp"
                    md_bg_color: 0.9, 0.92, 0.95, 1

                # Danh sách nút chọn Menu
                MDNavigationDrawerMenu:
                    
                    MDNavigationDrawerItem:
                        icon: "home-outline"
                        text: "Trang chủ"
                        selected_color: 0.12, 0.53, 0.68, 1
                        text_color: 0.2, 0.3, 0.4, 1
                        icon_color: 0.12, 0.53, 0.68, 1
                        focus_color: 0.88, 0.94, 0.97, 1 
                        radius: [16, ] 
                        on_release: nav_drawer.set_state("close")
                    
                    MDNavigationDrawerItem:
                        icon: "cog-outline"
                        text: "Cài đặt ứng dụng"
                        text_color: 0.2, 0.3, 0.4, 1
                        icon_color: 0.4, 0.5, 0.6, 1
                        focus_color: 0.88, 0.94, 0.97, 1
                        radius: [16, ]
                        on_release: nav_drawer.set_state("close")
                        
                    MDNavigationDrawerItem:
                        icon: "phone-outline"
                        text: "Liên hệ hỗ trợ"
                        text_color: 0.2, 0.3, 0.4, 1
                        icon_color: 0.4, 0.5, 0.6, 1
                        focus_color: 0.88, 0.94, 0.97, 1
                        radius: [16, ]
                        on_release: nav_drawer.set_state("close")
                    
                    MDNavigationDrawerItem:
                        icon: "power"
                        text: "Thoát"
                        text_color: 0.8, 0.3, 0.3, 1
                        icon_color: 0.8, 0.3, 0.3, 1
                        focus_color: 0.98, 0.9, 0.9, 1
                        radius: [16, ]
                        on_release: app.stop()
'''

class VSLAppScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        
        self.is_translating = False
        self.frames_buffer = []
        self.vote_buffer = deque(maxlen=config.VOTE_WINDOW)
        self.no_hand_counter = 0

        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"Lỗi khởi tạo pygame.mixer: {e}")
        self.last_spoken_text = ""  

        try:
            self.model = tf.keras.models.load_model(config.MODEL_PATH)
            with open(config.NAMES_PATH, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
            print("✓ Tải mô hình AI thành công!")
        except Exception as e:
            self.ids.result_label.text = "LỖI MODEL"
            print(f"Lỗi tải mô hình: {e}")

    def speak_text(self, text):
        try:
            tts = gTTS(text=text, lang='vi')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Lỗi phát âm thanh: {e}")

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def toggle_translation(self):
        self.is_translating = not self.is_translating
        
        if self.is_translating:
            self.ids.translate_btn.md_bg_color = (0.85, 0.32, 0.32, 1)  
            self.ids.translate_btn.text = "DỪNG NHẬN DIỆN"
            self.ids.status_tag.text = "Đang quét..."
            self.ids.result_label.text = "Đang chờ cử chỉ"
            self.ids.result_label.theme_text_color = "Secondary"
            self.frames_buffer.clear()
            self.vote_buffer.clear()
            self.last_spoken_text = "" 
        else:
            self.ids.translate_btn.md_bg_color = (0.12, 0.53, 0.68, 1)  
            self.ids.translate_btn.text = "BẮT ĐẦU DỊCH"
            self.ids.status_tag.text = "Hệ thống sẵn sàng"
            self.ids.result_label.text = "..."

    def extract_keypoints(self, results):
        data = np.zeros(126, dtype=np.float32)
        if results.multi_hand_landmarks:
            for i, hand_lm in enumerate(results.multi_hand_landmarks):
                if i >= 2: break
                hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32).flatten()
                data[i * 63 : (i + 1) * 63] = hand_data
        return data

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        hand_detected = bool(results.multi_hand_landmarks)

        if hand_detected:
            self.no_hand_counter = 0
        else:
            self.no_hand_counter += 1

        if self.is_translating:
            if self.no_hand_counter >= config.NO_HAND_RESET_LIMIT:
                self.frames_buffer.clear()
                self.vote_buffer.clear()
                self.ids.status_tag.text = "Đang quét..."
                self.ids.result_label.text = "Đang chờ cử chỉ"
                self.ids.result_label.theme_text_color = "Secondary"
                self.last_spoken_text = ""
                self.no_hand_counter = 0

            if hand_detected:
                kp = self.extract_keypoints(results)
                self.frames_buffer.append(kp)

                if len(self.frames_buffer) >= config.MAX_FRAMES:
                    seq = np.expand_dims(np.array(self.frames_buffer, dtype=np.float32), axis=0)
                    last_probs = self.model.predict(seq, verbose=0)[0]
                    pred_idx = int(np.argmax(last_probs))
                    confidence = float(last_probs[pred_idx])

                    self.vote_buffer.append(pred_idx)
                    if len(self.vote_buffer) >= config.VOTE_MIN_AGREE:
                        counts = Counter(self.vote_buffer)
                        winner_idx, winner_votes = counts.most_common(1)[0]

                        if winner_votes >= config.VOTE_MIN_AGREE and confidence >= config.CONFIDENCE_THRESHOLD:
                            label = self.class_names[winner_idx]
                            
                            self.ids.status_tag.text = "Khớp thành công"
                            self.ids.result_label.text = label
                            self.ids.result_label.theme_text_color = "Custom"
                            self.ids.result_label.text_color = (0.05, 0.62, 0.43, 1) 
                            
                            if label != self.last_spoken_text:
                                self.last_spoken_text = label
                                threading.Thread(target=self.speak_text, args=(label,), daemon=True).start()
                            
                        elif confidence < config.CONFIDENCE_THRESHOLD:
                            self.ids.status_tag.text = "Độ tin cậy thấp"
                            self.ids.result_label.text = "Hãy làm rõ nét hơn"
                            self.ids.result_label.theme_text_color = "Custom"
                            self.ids.result_label.text_color = (0.88, 0.52, 0.15, 1)
                            self.last_spoken_text = ""

                    self.frames_buffer.pop(0)

        frame_rgb_kivy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer = cv2.flip(frame_rgb_kivy, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
        self.ids.camera_feed.texture = texture

class VN_SignLanguageApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Teal" 
        
        Builder.load_string(KV)
        screen = VSLAppScreen()
        screen.start_camera()
        return screen

if __name__ == '__main__':
    VN_SignLanguageApp().run()