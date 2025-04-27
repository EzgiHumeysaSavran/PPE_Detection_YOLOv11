# 🛡️ PPE Detection using YOLOv11 🦺👷‍♀️

PPE Detection System is a Python-based application that enables real-time detection of Personal Protective Equipment (PPE) such as helmets and vests in camera feeds.
The system uses a YOLOv11 model for object detection and provides a Tkinter-based GUI for easy monitoring of multiple cameras simultaneously.

# ✨ Key Highlights
- Real-time detection of people, helmets, and vests

- Multi-camera support with threading for smooth performance

- Visual alerts for missing safety equipment (helmet, vest, or both)

- Sound notifications for detected violations

- User-friendly graphical interface built with Tkinter

- Modular code structure for future extensions (e.g., database logging, email notifications)

# 🛠️ How It Works
- Live video feeds from connected cameras are processed frame by frame.

- The YOLOv11 model predicts bounding boxes and classifies detected objects.

- The system checks whether detected persons are wearing helmets and vests.

- Alerts are triggered if PPE requirements are not met:

- Visual warnings are displayed directly on the video frame.

- Sound alarms are played for missing helmet, vest, or both.

- The system architecture is designed to support easy scaling to additional cameras or detection classes if needed.

# 📂 Datasets
The detection model was fine-tuned on a custom PPE datasets

- helmet_person_dataset

- PPE_Detection.v2i.yolov11 dataset

- Worker-Safety dataset

Model training was performed using the Ultralytics YOLOv11 framework.

# 🤝 Contributors
- @EzgiHumeysaSavran

- @tancperin