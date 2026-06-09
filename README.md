# SignBridge - Nhận Diện Ngôn Ngữ Ký Hiệu

## 1. Tổng quan dự án (Overview)
Rào cản giao tiếp giữa người khiếm thính và cộng đồng vẫn còn rất lớn do thiếu các công cụ hỗ trợ dịch thuật tức thời. Vì thế, dự án **SignBridge** của nhóm Mixi đã ra đời nhằm giải quyết vấn đề trên. 

Dự án ứng dụng Computer Vision và Deep Learning (mạng LSTM) để phát hiện cử chỉ tay thông qua webcam và dịch trực tiếp sang văn bản tiếng Việt theo thời gian thực, giúp kết nối cộng đồng người khiếm thính với xã hội một cách dễ dàng và tự nhiên hơn.

## 2. Hướng dẫn cài đặt và chạy chương trình (Installation & Usage)

### Yêu cầu hệ thống
* **Python:** Phiên bản 3.11

### Cài đặt thư viện
Bạn cần cài đặt các thư viện sau (có thể chạy lệnh dưới đây trong terminal):
```
pip install tensorflow mediapipe opencv-python numpy pandas scikit-learn Pillow
```

## 3. Các bước chạy chương trình
### Bước 1: Clone repository về máy tính của bạn:
```
git clone https://github.com/quan07-aiot/SignBridge_group_mixi.git
cd SignBridge_group_mixi 
```
### Bước 2: (ĐẶC BIỆT LƯU Ý) Mở file config.py và sửa đường dẫn của 2 biến VIDEO_DIR và LABEL_FILE thành đúng địa chỉ thư mục Dataset chứa video và file CSV trên máy tính cá nhân của bạn.
Ví dụ:
```
VIDEO_DIR  = r"C:\Users\Home\OneDrive\VSL\Dataset\Video"
LABEL_FILE = r"C:\Users\Home\OneDrive\VSL\Dataset\Label\Label.csv"
```
### Bước 3: Huấn luyện mô hình AI
Chạy file training.py để trích xuất đặc trưng tọa độ tay từ video dataset và tiến hành train mạng AI.
```
python training.py
```
### Bước 4: Nhận diện qua App
Sau khi mô hình huấn luyện xong, chạy file app1.py để bật camera và bắt đầu quá trình nhận diện cử chỉ theo thời gian thực.
```
python app1.py
```
## 4. Các cử chỉ hiện có (Available Gestures)
Hệ thống hiện tại đang được huấn luyện và có khả năng nhận diện các cử chỉ giao tiếp cơ bản sau:

👋 Xin Chào

👋 Tạm Biệt

🙏 Cảm ơn

## 5. Kế hoạch trong tương lai (Future Plans)
🔮 Future Scope: Mở rộng phạm vi từ vựng, thêm nhiều cử chỉ giao tiếp phức tạp hơn vào bộ dữ liệu huấn luyện.

🌐 Multilingual Support: Hỗ trợ dịch từ ngôn ngữ ký hiệu sang nhiều ngôn ngữ văn bản khác nhau ở nhiều vùng/ khu vực ở Việt Nam

📱 Mobile Application version: Phát triển thành ứng dụng trên thiết bị di động (Android/iOS) để tăng tính tiện dụng cho người dùng.

🧠 Improved AI Model: Cải tiến và tối ưu hóa kiến trúc mạng AI nhằm tăng độ chính xác và tốc độ xử lý khung hình.

## 6. Người đóng góp (Contributors)
Hoàng Mạnh Quân (https://github.com/quan07-aiot)   

Phạm Mạnh Thái (https://github.com/mthai-p)

Đỗ Thành Long (https://github.com/dtlonginhy)

Trịnh Đăng Khôi

Nhóm Mixi


