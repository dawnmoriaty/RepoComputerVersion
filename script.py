import os
import shutil
import random


# Đường dẫn gốc của dataset hiện tại
data_root = r"D:\Fullstack\script\data\dataset_for_train_model_cassava_leaf_with_yolov8_and_object_detection"

# Đường dẫn tới các thư mục nguồn
train_images_dir = os.path.join(data_root, "train", "images")
train_labels_dir = os.path.join(data_root, "train", "labels")
val_images_dir = os.path.join(data_root, "val", "images")
val_labels_dir = os.path.join(data_root, "val", "labels")

# Kiểm tra sự tồn tại của các thư mục
for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    if not os.path.exists(dir_path):
        print(f"Lỗi: Thư mục {dir_path} không tồn tại!")
        exit()
    else:
        print(f"Tìm thấy thư mục: {dir_path}")

# Tạo thư mục mới với cấu trúc mong muốn
output_dir = "selected_data"
os.makedirs(output_dir, exist_ok=True)

# Tạo các thư mục images và labels, sau đó tạo train/val bên trong
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
for main_dir in [images_dir, labels_dir]:
    os.makedirs(os.path.join(main_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(main_dir, "val"), exist_ok=True)

# Hàm để lấy danh sách cặp ảnh-nhãn
def get_image_label_pairs(images_dir, labels_dir):
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Tìm thấy {len(images)} file ảnh trong {images_dir}")
    pairs = []
    for img in images:
        label_file = img.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            pairs.append((img, label_file))
        else:
            print(f"Không tìm thấy nhãn cho {img} trong {labels_dir}")
    print(f"Tìm thấy {len(pairs)} cặp ảnh-nhãn hợp lệ trong {images_dir}")
    return pairs

# Lấy cặp ảnh-nhãn từ train và val
train_pairs = get_image_label_pairs(train_images_dir, train_labels_dir)
val_pairs = get_image_label_pairs(val_images_dir, val_labels_dir)

# Lấy ngẫu nhiên 4000 cặp cho train
selected_train = random.sample(train_pairs, min(3000, len(train_pairs)))

# Sao chép file cho train
for img, lbl in selected_train:
    shutil.copy(os.path.join(train_images_dir, img), os.path.join(images_dir, "train", img))
    shutil.copy(os.path.join(train_labels_dir, lbl), os.path.join(labels_dir, "train", lbl))

# Lấy ngẫu nhiên 1200 cặp cho val
selected_val = random.sample(val_pairs, min(750, len(val_pairs)))

# Sao chép file cho val
for img, lbl in selected_val:
    shutil.copy(os.path.join(val_images_dir, img), os.path.join(images_dir, "val", img))
    shutil.copy(os.path.join(val_labels_dir, lbl), os.path.join(labels_dir, "val", lbl))

# Tạo file data.yaml
yaml_content = {
    'train': './selected_data/images/train',
    'val': './selected_data/images/val',
    'nc': 5,  # Thay bằng số lượng lớp thực tế
    'names': ['healthy', 'bacterial_blight', 'brown_streak', 'green_mottle', 'mosaic_disease']  # Thay bằng tên lớp thực tế
}



print(f"Đã chọn {len(selected_train)} ảnh cho train và {len(selected_val)} ảnh cho val")
print(f"Được lưu tại: {output_dir}")
print("Đã tạo file data.yaml")