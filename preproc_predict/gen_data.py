import os
import json
import multiprocessing
from pathlib import Path
import laspy
import numpy as np
import subprocess
import torch

# Загрузка конфигурации
def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)


# Нарезка файлов
def process_file_cut_tiles(filename, input_directory, output_directory, tile_size=64):
    """
    Функция для нарезки одного файла .las с помощью lastile.
    """
    input_file = os.path.join(input_directory, filename)
    output_subdir = os.path.join(output_directory, os.path.splitext(filename)[0])
    output_subdir_f = os.path.join(output_directory,
                                   os.path.splitext(filename)[0],
                                   os.path.splitext(filename)[0])

    # Создаем подкаталог для текущего файла, если его нет
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    # Формируем команду для lastile
    command = [
        'lastile',
        '-i', input_file,
        '-tile_size', str(tile_size),
        '-o', output_subdir_f,
    ]

    # Выполняем команду
    subprocess.run(command)
    print(f"{filename} успешно нарезан и сохранен в {output_subdir}")


# Фильтрация
def filter_las_file(filepath, save_filepath):
    print(f'Проверка внутри функции file:{filepath}')
    print(f'Проверка внутри функции save:{save_filepath}')
    # Шаг 1: Чтение LAS-файла
    las_file = laspy.read(filepath)

    # Шаг 2: Извлечение координат Z (высоты)
    z_coordinates = las_file.z

    # Шаг 3: Вычисление среднего и стандартного отклонения
    mean_z = np.mean(z_coordinates)
    std_z = np.std(z_coordinates)

    # Шаг 4: Фильтрация точек на основе 3 сигм
    mask = (z_coordinates >= mean_z - 2 * std_z) & (z_coordinates <= mean_z + 2 * std_z)

    # Применение маски к остальным координатам
    las_file.points = las_file.points[mask]

    # Шаг 5: Перезапись очищенного файла под тем же именем
    las_file.write(save_filepath)
    print(f"Файл {save_filepath} успешно очищен и записан.")


# Нормализация
def normalize_tile(file_path, metadata_path, pt_path, target_points):
    with laspy.open(file_path) as las:
        points = las.read()

        # Предположим, что points.x, points.y, points.z - это объекты типа ScaledArrayView
        center = np.array([np.mean(np.array(points.x)),
                           np.mean(np.array(points.y)),
                           np.mean(np.array(points.z))])

        # Смещение центра
        points.x -= center[0]
        points.y -= center[1]
        points.z -= center[2]

        # Сэмплирование до target_points
        if len(points.x) > target_points:
            indices = np.random.choice(len(points.x), target_points, replace=False)
        else:
            indices = np.random.choice(len(points.x), target_points, replace=True)

        # Выборка атрибутов
        sampled_points = points[indices]
        sampled_rgb = np.column_stack((sampled_points.red,
                                       sampled_points.green,
                                       sampled_points.blue))

        # Сохранение метаданных
        metadata = {
            "filename": Path(file_path).name,
            "centroid": center.tolist()
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Преобразуем данные в тензоры PyTorch
        points_tensor = torch.tensor(
            np.column_stack((
                sampled_points.x,
                sampled_points.y,
                sampled_points.z,
                sampled_rgb
            )),
            dtype=torch.float32
        )

        # Сохраняем тензор в файл .pt
        torch.save(points_tensor, pt_path)

def normalize_and_save(file_path, metadata_path, pt_path, target_points):
    normalize_tile(file_path, metadata_path, pt_path, target_points)


# Параллельная обработка
def process_files(file_list, config):
    input_dir = config["input_directory_las"]
    output_cut = config["output_directory_cut_las"]
    output_filter = config["output_directory_filter_las"]
    output_pt = config["output_directory_pt"]
    metadata_dir = config["shift_metadata_directory"]
    tile_size = config["tile_size"]
    target_points = config["target_points"]

    # Создание необходимых каталогов
    os.makedirs(output_cut, exist_ok=True)
    os.makedirs(output_filter, exist_ok=True)
    os.makedirs(output_pt, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    dir_las_names = [os.path.splitext(filename)[0] for filename in file_list]

    # 1. Нарезка
    with multiprocessing.Pool() as pool:
        pool.starmap(process_file_cut_tiles, [(filename, input_dir, output_cut, tile_size) for filename in file_list])

    # 2. Фильтрация (если включено)
    if config["filtering_enabled"]:
        for dir_las_file_name in dir_las_names:
            dir_to_filter_files = os.path.join(output_filter,dir_las_file_name)
            print(f'Проверка: {dir_to_filter_files}')
            os.makedirs(dir_to_filter_files, exist_ok=True)
            las_files = list(Path(os.path.join(output_cut, dir_las_file_name)).rglob("*.las"))
            print(f'las_file:{las_files}')
            print(f'dic:{os.path.join(output_cut, dir_las_file_name)}')
            output_files = [os.path.join(dir_to_filter_files,las_file.name) for las_file in las_files]
            input_files = [os.path.join(output_cut, dir_las_file_name,las_file.name) for las_file in las_files]
            # Генерация списка кортежей для starmap
            args = [(input_file, output_file) for input_file, output_file in zip(input_files, output_files)]
            with multiprocessing.Pool() as pool:
                print(f'args={args}')
                pool.starmap(filter_las_file, args)

    # 3. Нормализация
    for dir_las_file_name in dir_las_names:
        dir_to_pt_files = os.path.join(output_pt, dir_las_file_name)
        dir_to_meta_files = os.path.join(metadata_dir, dir_las_file_name)
        os.makedirs(dir_to_pt_files, exist_ok=True)
        os.makedirs(dir_to_meta_files, exist_ok=True)
        with multiprocessing.Pool() as pool:
            pool.starmap(normalize_and_save,
                         [(filename,
                           os.path.join(dir_to_meta_files, Path(filename.name).with_suffix(".json")),
                           os.path.join(dir_to_pt_files, Path(filename.name).with_suffix(".pt")),
                           target_points)
                           for filename in
                             Path(output_filter, dir_las_file_name).rglob("*.las")])

# Главный запуск
if __name__ == "__main__":
    config = load_config()
    input_dir = config["input_directory_las"]
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.las')]
    process_files(input_files, config)
