import os
import random
import shutil


def train_val_split(input_dir, output_dir, split_ratio):
    files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
    random.shuffle(files)

    split_point = int(len(files) * split_ratio)
    train_files = files[:split_point]
    val_files = files[split_point:]

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(input_dir, f), os.path.join(output_dir, "train", f))
    for f in val_files:
        shutil.copy(os.path.join(input_dir, f), os.path.join(output_dir, "val", f))

    print("Данные разделены на train и val.")
