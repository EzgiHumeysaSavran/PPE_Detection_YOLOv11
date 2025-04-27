from ultralytics import YOLO

if __name__ == '__main__':
    print("Model yükleme başlıyor.")
    model = YOLO(r'..\runs\detect\train\weights\best.pt')
    print("Model başarıyla yüklendi.")

    print("Eğitim başlıyor.")
    model.train(
        data=r'..\PPE_Detection.v2i.yolov11\data.yaml',
        epochs=20,
        imgsz=640,
        batch=16,
)
    print("Eğitim tamamlandı.")

    print("Model kaydediliyor.")
    model.save('fine_tuned_model_v2.pt')
    print("Model başarıyla kaydedildi.")
