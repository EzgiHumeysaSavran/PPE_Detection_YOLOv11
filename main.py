import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
from queue import Queue
import winsound
import time

model = YOLO(r'..\runs\detect\train5\weights\best.pt')

last_alert_time = {}

def play_sound(alert_type, frequency, duration):
    current_time = time.time()
    if alert_type not in last_alert_time or (current_time - last_alert_time[alert_type] > 3):
        last_alert_time[alert_type] = current_time
        threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

def process_camera(cam_id, frame_queue):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        frame_queue.put((cam_id, None))
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put((cam_id, None))
            break

        results = model.predict(frame, conf=0.65, save=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes else []
        classes = results[0].boxes.cls.cpu().numpy() if results and results[0].boxes else []

        for i, box in enumerate(boxes):
            class_name = model.names[int(classes[i])]
            if class_name == "person":
                x1, y1, x2, y2 = map(int, box)
                helmet_detected = False
                vest_detected = False

                for j, other_box in enumerate(boxes):
                    other_class = model.names[int(classes[j])]
                    ox1, oy1, ox2, oy2 = map(int, other_box)
                    if x1 < ox1 < x2 and y1 < oy1 < y2:
                        if other_class == "helmet":
                            helmet_detected = True
                            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
                        elif other_class == "vest":
                            vest_detected = True
                            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)

                alert_message = ""
                if not helmet_detected and not vest_detected:
                    alert_message = "Uyari: no helmet, no vest"
                    color = (0, 0, 255)
                    play_sound("no_helmet_no_vest", 1000, 500)
                elif not helmet_detected:
                    alert_message = "Uyari: no helmet"
                    color = (0, 0, 255)
                    play_sound("no_helmet", 1200, 500)
                elif not vest_detected:
                    alert_message = "Uyari: no vest"
                    color = (0, 0, 255)
                    play_sound("no_vest", 800, 500)
                else:
                    color = (0, 255, 0)

                text_position = (x1 + 10, y1 - 10) if y1 > 20 else (x1 + 10, y1 + 20)
                if alert_message:
                    cv2.putText(frame, alert_message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        frame = cv2.resize(frame, (500, 500))
        frame_queue.put((cam_id, frame))

    cap.release()

def update_frame(frame_queue, display_label1, display_label2):
    if not frame_queue.empty():
        cam_id, frame = frame_queue.get()
        if frame is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)

            if cam_id == 0:
                display_label1.config(image=img_tk)
                display_label1.image = img_tk
            elif cam_id == 1:
                display_label2.config(image=img_tk)
                display_label2.image = img_tk

    app.after(10, update_frame, frame_queue, display_label1, display_label2)

def start_detection():
    global thread1, thread2
    camera1_label.config(text="Kamera 1: Başlatılıyor...", bg="black", fg="white")
    camera2_label.config(text="Kamera 2: Başlatılıyor...", bg="black", fg="white")

    frame_queue = Queue()

    thread1 = threading.Thread(target=process_camera, args=(0, frame_queue), daemon=True)
    thread2 = threading.Thread(target=process_camera, args=(1, frame_queue), daemon=True)
    thread1.start()
    thread2.start()

    update_frame(frame_queue, camera1_label, camera2_label)

def on_closing():
    if messagebox.askokcancel("Çıkış", "Kapatmak ister misiniz?"):
        app.quit()
        app.destroy()
        cv2.destroyAllWindows()

def update_status():
    camera1_label.config(text="Durum: Kamera 1 Aktif", width=50, height=50, bg="black", fg="white")
    camera2_label.config(text="Durum: Kamera 2 Aktif", width=50, height=50, bg="black", fg="white")

app = tk.Tk()
app.title("PPE Detection System")
app.geometry("1200x800")
app.configure(bg="lightgray")

start_button = tk.Button(
    app, text="Algılamayı Başlat", command=start_detection,
    height=2, width=20, bg="lightblue", fg="black", font=("Arial", 12, "bold")
)
start_button.grid(row=0, column=0, columnspan=2, pady=20)

app.protocol("WM_DELETE_WINDOW", on_closing)

status_button = tk.Button(
    app, text="Kamera Kontrol", command=update_status,
    height=2, width=15, bg="lightgreen", fg="black", font=("Arial", 10, "bold")
)
status_button.grid(row=3, column=0, columnspan=2, pady=10)

camera1_label = tk.Label(app, text="Kamera 1: Bekliyor...", bg="white", fg="black", font=("Helvetica", 10))
camera1_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

camera2_label = tk.Label(app, text="Kamera 2: Bekliyor...", bg="white", fg="black", font=("Helvetica", 10))
camera2_label.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

app.grid_rowconfigure(1, weight=1)
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)

app.mainloop()