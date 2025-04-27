import os

class_mapping = {
    4: 0,  # Protective Helmet → helmet
    5: 1   # Safety Vest → vest
}
ignored_classes = [0, 1, 2, 3, 6]  # Dust Mask, Eye Wear, Glove, Protective Boots, Shield

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

        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox_info = " ".join(parts[1:])

            if class_id in ignored_classes:
                continue

            new_class_id = class_mapping.get(class_id)
            if new_class_id is not None:
                updated_lines.append(f"{new_class_id} {bbox_info}\n")

        with open(label_path, 'w') as f:
            f.writelines(updated_lines)

print("Etiket dosyaları başarıyla güncellendi!")
