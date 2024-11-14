import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Buka webcam (gunakan indeks 0 untuk webcam default)
cap = cv2.VideoCapture("sholat.mp4")

# Pastikan webcam terbuka dengan baik
if not cap.isOpened():
    print("Error: Tidak dapat membuka webcam.")
    exit()

# Fungsi untuk mendeteksi gerakan sholat berdasarkan posisi landmark
def detect_prayer_pose(landmarks):
    if landmarks is None:
        return "Posisi Tidak Dikenal"

    # Mendapatkan koordinat penting (misalnya, pinggul, lutut, bahu, dll.)
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    head = landmarks[mp_pose.PoseLandmark.NOSE]
    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

    # Deteksi posisi berdiri (Qiyam)
    if abs(hip.y - knee.y) < 0.05 and abs(shoulder.y - head.y) < 0.05:
        return "Berdiri (Qiyam)"

    # Deteksi posisi ruku
    elif hip.y < knee.y and abs(shoulder.y - head.y) > 0.2:
        return "Ruku"

    # Deteksi posisi sujud
    elif head.y < knee.y and abs(hip.y - knee.y) < 0.1:
        return "Sujud"

    # Deteksi posisi duduk
    elif abs(hip.y - knee.y) > 0.2 and abs(shoulder.y - head.y) < 0.1:
        return "Duduk"

    return "Posisi Tidak Dikenal"

# Proses deteksi pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        
        # Jika frame gagal dibaca, berhenti
        if not ret:
            print("Error: Gagal membaca frame.")
            break
        
        # Resize frame untuk mempercepat proses
        frame = cv2.resize(frame, (640, 480))  # Ubah ukuran frame menjadi 640x480

        # Konversi frame menjadi RGB karena MediaPipe memerlukan format RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set image menjadi tidak dapat ditulis untuk efisiensi
        image.flags.writeable = False
        results = pose.process(image)

        # Konversi kembali ke format BGR untuk OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Gambar landmarks pose jika ada
        if results.pose_landmarks:
            # Gambar landmarks tubuh
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Deteksi gerakan sholat
            pose_text = detect_prayer_pose(results.pose_landmarks.landmark)
            
            # Menambahkan teks di atas bounding box
            cv2.putText(image, pose_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Menghitung bounding box berdasarkan koordinat landmark (kiri atas dan kanan bawah)
            landmarks = results.pose_landmarks.landmark
            x_min = min([landmarks[i].x for i in range(len(landmarks))]) * frame.shape[1]
            y_min = min([landmarks[i].y for i in range(len(landmarks))]) * frame.shape[0]
            x_max = max([landmarks[i].x for i in range(len(landmarks))]) * frame.shape[1]
            y_max = max([landmarks[i].y for i in range(len(landmarks))]) * frame.shape[0]

            # Gambar bounding box di sekitar tubuh
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        # Tampilkan frame yang sudah diproses
        cv2.imshow("Pose Detection - Sholat", image)

        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Lepaskan objek capture dan tutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()
