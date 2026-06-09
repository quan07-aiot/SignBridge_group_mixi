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
import io  # <-- Thêm thư viện io để đọc/ghi trực tiếp trên RAM
# -------------------------------

from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen

# Đọc cấu hình từ file config.py của bạn
import config

# Khởi tạo MediaPipe theo đúng cấu hình
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=config.MAX_NUM_HANDS,
    min_detection_confidence=config.MIN_DETECTION_CONF,
    min_tracking_confidence=config.MIN_TRACKING_CONF
)

# GIAO DIỆN ỨNG DỤNG
KV = '''
<VSLAppScreen>:
    MDNavigationLayout:
        MDScreenManager:
            MDScreen:
                MDBoxLayout:
                    orientation: 'vertical'
                    
                    MDTopAppBar:
                        title: "Ngôn ngữ ký hiệu tiếng Việt"
                        elevation: 4
                        left_action_items: [["menu", lambda x: nav_drawer.set_state("open")]]
                        md_bg_color: 0.12, 0.15, 0.28, 1
                    
                    MDBoxLayout:
                        orientation: 'vertical'
                        padding: "10dp"
                        spacing: "10dp"
                        
                        # Ô VĂN BẢN HIỂN THỊ KẾT QUẢ
                        MDCard:
                            size_hint: 1, 0.15
                            elevation: 2
                            padding: "5dp"
                            md_bg_color: 0.95, 0.95, 0.95, 1
                            orientation: "vertical"
                            
                            # DÒNG 1: Chữ to để hiện kết quả dịch
                            MDLabel:
                                id: result_label
                                text: "Đang chờ hành động..."
                                halign: "center"
                                font_style: "H5"
                                theme_text_color: "Primary"
                                
                            # DÒNG 2: Chữ nhỏ để chạy vòng lặp 1/20
                            MDLabel:
                                id: progress_label
                                text: "Tiến trình: 0/20"
                                halign: "center"
                                font_style: "Caption"
                                theme_text_color: "Secondary"

                        # CAMERA TRUNG TÂM
                        MDCard:
                            size_hint: 1, 0.65
                            elevation: 3
                            radius: [20, ]
                            overflow: 'hidden'
                            Image:
                                id: camera_feed
                                allow_stretch: True
                                keep_ratio: True

                        # NÚT DỊCH BÊN DƯỚI
                        MDBoxLayout:
                            size_hint: 1, 0.2
                            padding: [0, "10dp", 0, 0]
                            MDRaisedButton:
                                id: translate_btn
                                text: "DỊCH NGÔN NGỮ KÝ HIỆU"
                                pos_hint: {"center_x": .5}
                                size_hint: 0.8, None
                                height: "56dp"
                                font_size: "18sp"
                                md_bg_color: 0.12, 0.45, 0.28, 1
                                on_release: root.toggle_translation()

        # THANH MENU (NAVIGATION DRAWER)
        MDNavigationDrawer:
            id: nav_drawer
            radius: (0, 16, 16, 0)
            MDNavigationDrawerMenu:
                MDNavigationDrawerHeader:
                    title: "Menu"
                    text: "Ứng dụng hỗ trợ VSL"
                    spacing: "4dp"
                    padding: "12dp", 0, 0, "56dp"
                
                MDNavigationDrawerItem:
                    icon: "home"
                    text: "Trang chủ"
                MDNavigationDrawerItem:
                    icon: "contacts"
                    text: "Liên hệ"
                MDNavigationDrawerItem:
                    icon: "login"
                    text: "Đăng nhập"
'''

class VSLAppScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        
        # Biến trạng thái logic
        self.is_translating = False
        self.frames_buffer = []
        self.vote_buffer = deque(maxlen=config.VOTE_WINDOW)
        self.no_hand_counter = 0

        # --- KHỞI TẠO ÂM THANH ---
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"Lỗi khởi tạo pygame.mixer: {e}")
        self.last_spoken_text = ""  # Lưu từ vừa đọc để tránh lặp âm liên tục
        # -------------------------

        # Tải mô hình AI và nhãn
        try:
            self.model = tf.keras.models.load_model(config.MODEL_PATH)
            with open(config.NAMES_PATH, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
            print("✓ Tải mô hình thành công lên App!")
        except Exception as e:
            self.ids.result_label.text = "LỖI: Không tải được Model!"
            print(f"Lỗi tải mô hình: {e}")

    def speak_text(self, text):
        """Hàm tạo và phát âm thanh tiếng Việt sử dụng RAM đệm"""
        try:
            # Tạo luồng byte trên RAM thay vì lưu ra ổ cứng
            tts = gTTS(text=text, lang='vi')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Tải và phát âm thanh trực tiếp từ RAM
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
            self.ids.translate_btn.md_bg_color = (0.8, 0.2, 0.2, 1)
            self.ids.translate_btn.text = "ĐANG NHẬN DIỆN..."
            self.frames_buffer.clear()
            self.vote_buffer.clear()
            self.last_spoken_text = "" # Reset âm thanh khi bật lại
        else:
            self.ids.translate_btn.md_bg_color = (0.12, 0.45, 0.28, 1)
            self.ids.translate_btn.text = "DỊCH NGÔN NGỮ KÝ HIỆU"

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

        # Vẽ xương tay lên camera
        if hand_detected:
            self.no_hand_counter = 0
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        else:
            self.no_hand_counter += 1

        # Xử lý logic AI nếu đang bật nút Dịch
        if self.is_translating:
            # Nếu mất tay quá lâu -> Xóa bộ đệm, quay về 0/20 và reset âm thanh
            if self.no_hand_counter >= config.NO_HAND_RESET_LIMIT:
                self.frames_buffer.clear()
                self.no_hand_counter = 0
                self.ids.progress_label.text = "Tiến trình: 0/20"
                self.last_spoken_text = "" # Cho phép đọc lại từ đầu khi tay xuất hiện lại

            if hand_detected:
                kp = self.extract_keypoints(results)
                self.frames_buffer.append(kp)
                
                # HIỂN THỊ VÒNG LẶP XUỐNG DÒNG PROGRESS
                self.ids.progress_label.text = f"Đang thu thập: {len(self.frames_buffer)}/{config.MAX_FRAMES}"

                # Khi đủ 20/20 khung hình
                if len(self.frames_buffer) >= config.MAX_FRAMES:
                    seq = np.expand_dims(np.array(self.frames_buffer, dtype=np.float32), axis=0)
                    last_probs = self.model.predict(seq, verbose=0)[0]
                    pred_idx = int(np.argmax(last_probs))
                    confidence = float(last_probs[pred_idx])

                    # Thuật toán Voting
                    self.vote_buffer.append(pred_idx)
                    if len(self.vote_buffer) >= config.VOTE_MIN_AGREE:
                        counts = Counter(self.vote_buffer)
                        winner_idx, winner_votes = counts.most_common(1)[0]

                        # NẾU DỰ ĐOÁN ĐÚNG -> HIỂN THỊ LÊN RESULT LABEL & PHÁT ÂM THANH
                        if winner_votes >= config.VOTE_MIN_AGREE and confidence >= config.CONFIDENCE_THRESHOLD:
                            label = self.class_names[winner_idx]
                            self.ids.result_label.text = f"{label} ({confidence*100:.1f}%)"
                            self.ids.result_label.theme_text_color = "Custom"
                            self.ids.result_label.text_color = (0, 0.6, 0.2, 1) # Chữ đổi sang Xanh lá
                            
                            # --- GỌI ÂM THANH ---
                            if label != self.last_spoken_text:
                                self.last_spoken_text = label
                                # Khởi chạy luồng ẩn để không làm đơ camera
                                threading.Thread(target=self.speak_text, args=(label,), daemon=True).start()
                            # --------------------
                            
                        # Nếu nhận diện kém -> Cảnh báo
                        elif confidence < config.CONFIDENCE_THRESHOLD:
                            self.ids.result_label.text = "Không rõ cử chỉ"
                            self.ids.result_label.theme_text_color = "Custom"
                            self.ids.result_label.text_color = (0.8, 0.4, 0, 1) # Chữ đổi sang Cam
                            
                            # --- RESET GIỌNG NÓI MỖI KHI MẤT NHẬN DIỆN ---
                            self.last_spoken_text = ""
                            # ---------------------------------------------

                    # Cửa sổ trượt
                    self.frames_buffer.pop(0)

        # Render ảnh lên UI Kivy (Đổi BGR sang RGB cho chuẩn Kivy Texture)
        frame_rgb_kivy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer = cv2.flip(frame_rgb_kivy, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
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

# Đọc cấu hình từ file config.py của bạn
import config

# Khởi tạo MediaPipe theo đúng cấu hình
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=config.MAX_NUM_HANDS,
    min_detection_confidence=config.MIN_DETECTION_CONF,
    min_tracking_confidence=config.MIN_TRACKING_CONF
)

# GIAO DIỆN ỨNG DỤNG
KV = '''
<VSLAppScreen>:
    MDNavigationLayout:
        MDScreenManager:
            MDScreen:
                MDBoxLayout:
                    orientation: 'vertical'
                    
                    MDTopAppBar:
                        title: "Ngôn ngữ ký hiệu tiếng Việt"
                        elevation: 4
                        left_action_items: [["menu", lambda x: nav_drawer.set_state("open")]]
                        md_bg_color: 0.12, 0.15, 0.28, 1
                    
                    MDBoxLayout:
                        orientation: 'vertical'
                        padding: "10dp"
                        spacing: "10dp"
                        
                        # Ô VĂN BẢN HIỂN THỊ KẾT QUẢ
                        MDCard:
                            size_hint: 1, 0.15
                            elevation: 2
                            padding: "5dp"
                            md_bg_color: 0.95, 0.95, 0.95, 1
                            orientation: "vertical"
                            
                            # DÒNG 1: Chữ to để hiện kết quả dịch
                            MDLabel:
                                id: result_label
                                text: "Hãy thực hiện hành động"
                                halign: "center"
                                font_style: "H5"
                                theme_text_color: "Primary"
                                
                            # DÒNG 2: Chữ nhỏ để chạy vòng lặp 1/20
                            MDLabel:
                                id: progress_label
                                text: "Tiến trình: 0/20"
                                halign: "center"
                                font_style: "Caption"
                                theme_text_color: "Secondary"

                        # CAMERA TRUNG TÂM
                        MDCard:
                            size_hint: 1, 0.65
                            elevation: 3
                            radius: [20, ]
                            overflow: 'hidden'
                            Image:
                                id: camera_feed
                                allow_stretch: True
                                keep_ratio: True

                        # NÚT DỊCH BÊN DƯỚI
                        MDBoxLayout:
                            size_hint: 1, 0.2
                            padding: [0, "10dp", 0, 0]
                            MDRaisedButton:
                                id: translate_btn
                                text: "DỊCH NGÔN NGỮ KÝ HIỆU"
                                pos_hint: {"center_x": .5}
                                size_hint: 0.8, None
                                height: "56dp"
                                font_size: "18sp"
                                md_bg_color: 0.12, 0.45, 0.28, 1
                                on_release: root.toggle_translation()

        # THANH MENU (NAVIGATION DRAWER)
        MDNavigationDrawer:
            id: nav_drawer
            radius: (0, 16, 16, 0)
            MDNavigationDrawerMenu:
                MDNavigationDrawerHeader:
                    title: "Menu"
                    text: "Ứng dụng hỗ trợ VSL"
                    spacing: "4dp"
                    padding: "12dp", 0, 0, "56dp"
                
                MDNavigationDrawerItem:
                    icon: "home"
                    text: "Trang chủ"
                MDNavigationDrawerItem:
                    icon: "contacts"
                    text: "Liên hệ"
                MDNavigationDrawerItem:
                    icon: "login"
                    text: "Đăng nhập"
'''

class VSLAppScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        
        # Biến trạng thái logic
        self.is_translating = False
        self.frames_buffer = []
        self.vote_buffer = deque(maxlen=config.VOTE_WINDOW)
        self.no_hand_counter = 0

        # --- KHỞI TẠO ÂM THANH ---
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"Lỗi khởi tạo pygame.mixer: {e}")
        self.last_spoken_text = ""  
        # -------------------------

        # Tải mô hình AI và nhãn
        try:
            self.model = tf.keras.models.load_model(config.MODEL_PATH)
            with open(config.NAMES_PATH, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
            print("✓ Tải mô hình thành công lên App!")
        except Exception as e:
            self.ids.result_label.text = "LỖI: Không tải được Model!"
            print(f"Lỗi tải mô hình: {e}")

    def speak_text(self, text):
        """Hàm tạo và phát âm thanh tiếng Việt sử dụng RAM đệm"""
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
            self.ids.translate_btn.md_bg_color = (0.8, 0.2, 0.2, 1)
            self.ids.translate_btn.text = "ĐANG NHẬN DIỆN..."
            self.frames_buffer.clear()
            self.vote_buffer.clear()
            self.ids.result_label.text = "Hãy thực hiện hành động"
            self.ids.result_label.theme_text_color = "Primary"
            self.last_spoken_text = "" 
        else:
            self.ids.translate_btn.md_bg_color = (0.12, 0.45, 0.28, 1)
            self.ids.translate_btn.text = "DỊCH NGÔN NGỮ KÝ HIỆU"

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

        # Vẽ xương tay lên camera
        if hand_detected:
            self.no_hand_counter = 0
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
        else:
            self.no_hand_counter += 1

        # Xử lý logic AI nếu đang bật nút Dịch
        if self.is_translating:
            # --- ĐÃ SỬA: Nếu mất tay quá lâu -> Reset giao diện về trạng thái CHỜ ---
            if self.no_hand_counter >= config.NO_HAND_RESET_LIMIT:
                self.frames_buffer.clear()
                self.vote_buffer.clear()
                self.ids.progress_label.text = "Tiến trình: 0/20"
                
                # Trả text hiển thị về trạng thái chờ mặc định
                self.ids.result_label.text = "Hãy thực hiện hành động"
                self.ids.result_label.theme_text_color = "Primary"
                
                self.last_spoken_text = "" # Reset bộ nhớ âm thanh để khi đưa tay lên lại vẫn đọc được cử chỉ cũ
                self.no_hand_counter = 0

            if hand_detected:
                kp = self.extract_keypoints(results)
                self.frames_buffer.append(kp)
                
                self.ids.progress_label.text = f"Đang thu thập: {len(self.frames_buffer)}/{config.MAX_FRAMES}"

                # Khi đủ 20/20 khung hình
                if len(self.frames_buffer) >= config.MAX_FRAMES:
                    seq = np.expand_dims(np.array(self.frames_buffer, dtype=np.float32), axis=0)
                    last_probs = self.model.predict(seq, verbose=0)[0]
                    pred_idx = int(np.argmax(last_probs))
                    confidence = float(last_probs[pred_idx])

                    # Thuật toán Voting
                    self.vote_buffer.append(pred_idx)
                    if len(self.vote_buffer) >= config.VOTE_MIN_AGREE:
                        counts = Counter(self.vote_buffer)
                        winner_idx, winner_votes = counts.most_common(1)[0]

                        # NẾU DỰ ĐOÁN ĐÚNG -> HIỂN THỊ LÊN RESULT LABEL & PHÁT ÂM THANH
                        if winner_votes >= config.VOTE_MIN_AGREE and confidence >= config.CONFIDENCE_THRESHOLD:
                            label = self.class_names[winner_idx]
                            self.ids.result_label.text = f"{label} ({confidence*100:.1f}%)"
                            self.ids.result_label.theme_text_color = "Custom"
                            self.ids.result_color = (0, 0.6, 0.2, 1)
                            self.ids.result_label.text_color = (0, 0.6, 0.2, 1)
                            
                            # --- GỌI ÂM THANH ---
                            if label != self.last_spoken_text:
                                self.last_spoken_text = label
                                threading.Thread(target=self.speak_text, args=(label,), daemon=True).start()
                            
                        # Nếu nhận diện kém -> Cảnh báo
                        elif confidence < config.CONFIDENCE_THRESHOLD:
                            self.ids.result_label.text = "Không rõ cử chỉ"
                            self.ids.result_label.theme_text_color = "Custom"
                            self.ids.result_label.text_color = (0.8, 0.4, 0, 1)
                            
                            self.last_spoken_text = ""

                    # Cửa sổ trượt
                    self.frames_buffer.pop(0)

        # Render ảnh lên UI Kivy
        frame_rgb_kivy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer = cv2.flip(frame_rgb_kivy, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
        self.ids.camera_feed.texture = texture

class VN_SignLanguageApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Indigo"
        Builder.load_string(KV)
        screen = VSLAppScreen()
        screen.start_camera()
        return screen

if __name__ == '__main__':
    VN_SignLanguageApp().run()