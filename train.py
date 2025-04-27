from ultralytics import YOLO

if __name__ == '__main__':
    print("Model yükleme başlıyor.")
    model = YOLO('yolo11n.pt')
    print("Model başarıyla yüklendi.")

    print("Eğitim başlıyor.")
    model.train(
        data=r'..\helmet_person_dataset\data.yaml',
        epochs=40,
        imgsz=640,
        batch=16
    )
    print("Eğitim tamamlandı.")

    print("Model kaydediliyor.")
    model.save('ppe_detector.pt')
    print("Model başarıyla kaydedildi.")

