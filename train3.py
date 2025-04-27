from ultralytics import YOLO

if __name__ == '__main__':
    print("Model yükleme başlıyor.")
    model = YOLO(r'..\runs\detect\train2\weights\best.pt')
    print("Model başarıyla yüklendi.")

    print("Eğitim başlıyor.")
    model.train(
    data=r'..\Worker-Safety\data.yaml', 
    epochs=50,
    imgsz=640,
    batch=16,
)
    print("Eğitim tamamlandı.")

    print("Model kaydediliyor.")
    model.save('fine_tuned_model_v3.2.pt')
    print("Model başarıyla kaydedildi.")