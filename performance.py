from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO(r'..\runs\detect\train5\weights\best.pt')

    metrics = model.val(data=r'..\Worker-Safety\data.yaml') 

    print("Validation Metrics For Third.Two Training With Worker_Safety Dataset:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall: {metrics.box.r.mean():.4f}")
