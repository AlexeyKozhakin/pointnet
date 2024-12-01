import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class PointCloudDataset(Dataset):
    """
    Класс для загрузки данных из .pt файлов с разделением на train/val.
    """

    def __init__(self, data_dir, split="train", split_ratio=0.8, num_points=4096, num_channels=6, transform=None):
        """
        Инициализация.

        :param data_dir: Директория с .pt файлами.
        :param split: Тип данных ("train" или "val").
        :param split_ratio: Доля данных для обучения (по умолчанию 0.8).
        :param num_points: Количество точек в облаке (N).
        :param num_channels: Количество каналов (например, x, y, z, r, g, b).
        :param transform: Опциональные преобразования для точек.
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.num_channels = num_channels
        self.transform = transform

        # Получаем список всех файлов
        self.pt_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]

        # Разделяем на train/val
        train_files, val_files = train_test_split(self.pt_files, train_size=split_ratio, random_state=42)
        self.pt_files = train_files if split == "train" else val_files

    def __len__(self):
        """
        Возвращает количество файлов .pt в датасете.
        """
        return len(self.pt_files)

    def __getitem__(self, idx):
        """
        Возвращает один батч из файла.

        :param idx: Индекс файла .pt.
        :return: Точки и классы.
        """
        pt_file = self.pt_files[idx]
        data = torch.load(pt_file)  # Загружаем данные из файла

        # data.shape = [B, num_points, num_channels]
        points = data[:, :, :self.num_channels]  # Например, x, y, z, r, g
        labels = data[:, :, self.num_channels]  # Например, classification

        if self.transform:
            points = self.transform(points)

        return points, labels


from torch.utils.data import DataLoader


def create_dataloader(data_dir, batch_size, split, split_ratio=0.8, num_points=4096, num_channels=6, shuffle=True,
                      num_workers=4):
    """
    Создает DataLoader для загрузки данных с разделением на train/val.

    :param data_dir: Директория с .pt файлами.
    :param batch_size: Размер батча.
    :param split: Тип данных ("train" или "val").
    :param split_ratio: Доля данных для обучения.
    :param num_points: Количество точек в облаке.
    :param num_channels: Количество каналов.
    :param shuffle: Перемешивать ли данные.
    :param num_workers: Количество потоков для загрузки.
    :return: DataLoader.
    """
    dataset = PointCloudDataset(data_dir, split=split, split_ratio=split_ratio, num_points=num_points,
                                num_channels=num_channels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        collate_fn=collate_fixed_shape
    )
    return dataloader


def collate_fixed_shape(batch):
    """
    Кастомная функция для объединения батчей.
    """
    points_batch = []
    labels_batch = []

    for points, labels in batch:
        points_batch.append(points)  # [B, num_points, num_channels-1]
        labels_batch.append(labels)  # [B, num_points]

    # Конкатенируем данные по первому измерению
    points_batch = torch.cat(points_batch, dim=0)  # [N*B, num_points, num_channels-1]
    labels_batch = torch.cat(labels_batch, dim=0)  # [N*B, num_points]

    return points_batch, labels_batch
