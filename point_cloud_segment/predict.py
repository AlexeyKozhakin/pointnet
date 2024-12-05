import os
import json
import torch
import numpy as np
from multiprocessing import Pool
from scipy.spatial import KDTree
import laspy
import importlib

# Загрузка модели
def load_model(config):
    # Подгружаем файл с определением модели (если оно в другом модуле)
    model_class = importlib.import_module("models.pointnet")  # Подставьте реальный модуль
    model = model_class.PointNet(num_classes=20)  # Подставьте реальное имя класса модели

    # Загрузка только весов
    model_path = os.path.join(config["model_path"], config["model_name"])
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Загрузка точек и смещений
def load_data(pt_file, json_file):
    points = torch.load(pt_file)  # Загрузка файла *.pt
    print(f'Размерность точек: {points.shape}')
    with open(json_file, "r") as f:
        offsets = json.load(f)  # Загрузка смещений
    return points, offsets

# Прогноз и сохранение
import os
import numpy as np
import torch
from laspy import LasData, open as las_open
from scipy.spatial import KDTree

def process_file(args):
    pt_file, json_file, las_file, output_directory, model = args
    base_name = os.path.splitext(os.path.basename(pt_file))[0]

    # Загрузка точек и мета-данных
    points, offsets = load_data(pt_file, json_file)
    print(f'Загруженные точки: {points}')
    points[:, :3] += np.array(offsets["centroid"])  # Смещение координат

    # Прогнозирование классов
    with torch.no_grad():
        # Преобразуем точки в тензор, убрав вызов `.copy()`:
        if isinstance(points, np.ndarray):
            points_tensor = torch.tensor(points[:, :6], dtype=torch.float32)
        elif isinstance(points, torch.Tensor):
            points_tensor = points[:, :6].clone().detach().float()  # Для тензоров используем clone().detach()

        predictions = model(points_tensor.unsqueeze(0)).argmax(dim=2).squeeze(0).numpy()
        print('classes:',np.unique(predictions))

    # Загрузка LAS-файла
    las_data = laspy.read(las_file)

    # Расширение предсказаний на все точки LAS-файла с использованием k-NN
    kdtree = KDTree(points[:, :3].numpy() if isinstance(points, torch.Tensor) else points[:, :3])
    all_points = np.vstack((las_data.x, las_data.y, las_data.z)).T
    _, indices = kdtree.query(all_points, k=1)
    las_classes = predictions[indices]

    # Добавление предсказанных классов в LAS-файл
    las_data.classification = las_classes.astype(np.uint8)

    # Сохранение результатов в новый LAS-файл
    output_file = os.path.join(output_directory, f"{base_name}_predicted.las")
    las_data.write(output_file)

    print(f"Файл сохранен: {output_file}")
    return output_file



# Основной процесс
def main():
    # Загрузка конфигурации
    with open("config.json", "r") as f:
        config = json.load(f)

    model = load_model(config)

    dirs = [f for f in os.listdir(config["files_pt"])]
    for dir in dirs:
        # Сбор файлов
        pt_files = [os.path.join(config["files_pt"], dir, f)
                    for f in os.listdir(os.path.join(config["files_pt"], dir))
                    if f.endswith(".pt")]
        json_files = [os.path.join(config["files_json"], dir, f)
                      for f in os.listdir(os.path.join(config["files_json"], dir))
                      if f.endswith(".json")]
        las_files = [os.path.join(config["las_files"], dir, f)
                     for f in os.listdir(os.path.join(config["las_files"], dir))
                     if f.endswith(".las")]

        print(f'pt_files: {pt_files}')
        print(f'json_files: {json_files}')
        print(f'las_files: {las_files}')

        # Создание каталога для вывода
        os.makedirs(config["output_directory"], exist_ok=True)

        # Список аргументов для параллельной обработки
        tasks = [(pt_files[i], json_files[i], las_files[i], config["output_directory"], model) for i in range(len(pt_files))]
        print(f'tasks: {tasks}')
        # Параллельная обработка
        with Pool() as pool:
            results = pool.map(process_file, tasks)

    print("Обработка завершена. Результаты сохранены в:", config["output_directory"])

if __name__ == "__main__":
    main()
