from ultralytics import YOLO
import glob
import os
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':

    model = YOLO(r'..\runs\detect\train24\weights\best.pt')

    results = model.predict(
        source=r'..\PPE_Detection.v2i.yolov11\test\images',
        conf=0.25,
        save=True
    )

    for result in results:
        result.plot()

    print("Tahmin işlemi tamamlandı.")

    latest_folder = max(glob.glob('runs/detect/predict*/'), key=os.path.getmtime)

    for img_path in glob.glob(f'{latest_folder}/*.jpg')[1:4]:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Görüntü: {os.path.basename(img_path)}")
        plt.show()
