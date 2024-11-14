import cv2
import os

# Muat cascade classifier untuk deteksi tubuh manusia
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Buka video
cap = cv2.VideoCapture("1")

# Pastikan video terbuka dengan baik
if not cap.isOpened():
    print("Error: Tidak dapat membuka video.")
    exit()

# Buat folder untuk menyimpan anotasi
if not os.path.exists('Anotasi'):
    os.makedirs('Anotasi')

frame_number = 0
while True:
    # Baca frame dari video
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_number += 1
    height, width, _ = frame.shape

    # Ubah frame menjadi grayscale untuk deteksi objek
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi tubuh manusia dalam frame
    bodies = body_cascade.detectMultiScale(gray, 1.1, 3)

    # Siapkan file anotasi untuk frame ini
    annotation_filename = f"annotations/frame_{frame_number}.txt"
    
    with open(annotation_filename, 'w') as file:
        for (x, y, w, h) in bodies:
            # Konversi koordinat ke format YOLO
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            bbox_width = w / width
            bbox_height = h / height

            # Tulis anotasi ke file
            file.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")
            
            # Gambar bounding box pada frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tampilkan frame dengan bounding box
    cv2.imshow("Frame with Annotations", frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan objek capture dan tutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()
