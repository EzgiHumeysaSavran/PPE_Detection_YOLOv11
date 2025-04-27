import os

yaml_classes = ['helmet', 'vest']
yaml_class_ids = range(len(yaml_classes))

labels_dirs = [
    r'..\PPE_Detection.v2i.yolov11\train\labels',
    r'..\PPE_Detection.v2i.yolov11\valid\labels',
    r'..\PPE_Detection.v2i.yolov11\test\labels'
]

for labels_dir in labels_dirs:
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Format hatası: {label_path}, satır {line_num}")
                continue

            class_id = int(parts[0])
            if class_id not in yaml_class_ids:
                print(f"Uyumsuz sınıf ID'si: {label_path}, satır {line_num}, sınıf ID'si: {class_id}")
