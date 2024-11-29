import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset import create_dataloader
from models.pointnet import PointNet
import sys

# Получаем абсолютный путь к каталогу utils
utils_path = os.path.join(os.path.dirname(__file__), "utils")

# Добавляем utils в пути поиска модулей
sys.path.append(utils_path)

def calculate_accuracy(outputs, labels, num_classes, selected_classes=None):
    """
    Рассчитывает точность (accuracy) для указанных классов.
    Args:
        outputs (torch.Tensor): Логиты (предсказания модели) [B * N, num_classes].
        labels (torch.Tensor): Истинные метки [B * N].
        num_classes (int): Общее количество классов.
        selected_classes (list[int]): Классы, для которых считать метрику. Если None, считаем для всех.
    Returns:
        accuracies (dict): Словарь с точностью для каждого класса.
    """
    selected_classes = selected_classes or list(range(num_classes))
    accuracies = {}
    preds = torch.argmax(outputs, dim=1)  # Предсказания модели [B * N]

    for cls in selected_classes:
        mask = labels == cls  # Маска для текущего класса
        total = mask.sum().item()
        if total > 0:  # Если есть элементы этого класса
            correct = (preds[mask] == labels[mask]).sum().item()
            accuracies[cls] = correct / total
        else:
            accuracies[cls] = 0.0  # Если элементов нет, точность 0
    return accuracies


def evaluate_model(model, data_loader, criterion, num_classes, device, selected_classes=None):
    """
    Оценивает модель на заданном наборе данных.
    Args:
        model (torch.nn.Module): Обученная модель.
        data_loader (DataLoader): Даталоадер с валидационными или тестовыми данными.
        criterion (nn.Module): Функция потерь.
        num_classes (int): Общее количество классов.
        device (str): Устройство для вычислений ('cpu' или 'cuda').
        selected_classes (list[int]): Классы, для которых считать метрику.
    Returns:
        avg_loss (float): Средний лосс на наборе данных.
        class_accuracies (dict): Точность для каждого класса.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for points, labels in data_loader:
            points = points.to(device).float()  # [B, N, C]
            labels = labels.to(device).long()  # [B, N]

            outputs = model(points)  # [B, N, num_classes]
            outputs = outputs.reshape(-1, num_classes)  # (B * N, num_classes)
            labels = labels.view(-1)  # (B * N)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # Умножаем на количество примеров
            total_samples += labels.size(0)

            all_outputs.append(outputs)
            all_labels.append(labels)

    avg_loss = total_loss / total_samples
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    class_accuracies = calculate_accuracy(all_outputs, all_labels, num_classes, selected_classes)
    return avg_loss, class_accuracies


if __name__ == "__main__":
    # Загружаем конфигурацию
    with open("../config.json", "r") as f:
        config = json.load(f)

    data_dir = config["data"]["output_dir"]
    batch_size = config["training"]["batch_size"]
    num_points = config["data"]["num_points"]
    num_channels = config["data"]["num_channels"]
    split_ratio = config["data"]["train_val_split"]
    selected_classes = config["metrics"]["selected_classes"]  # Список классов для метрик

    # Создаем DataLoader для train и val
    train_loader = create_dataloader(data_dir, batch_size, split="train", split_ratio=split_ratio, num_points=num_points, num_channels=num_channels, shuffle=True)
    val_loader = create_dataloader(data_dir, batch_size, split="val", split_ratio=split_ratio, num_points=num_points, num_channels=num_channels, shuffle=False)

    for points, labels in train_loader:
        print(points.shape, labels.shape)

    model = PointNet(num_classes=config["training"]["num_classes"]).to(config["training"]["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["training"]["num_epochs"]):
        # Тренировка
        model.train()
        epoch_loss = 0.0
        all_train_outputs = []
        all_train_labels = []

        for points, labels in train_loader:
            points = points.to(config["training"]["device"]).float()  # [B, N, C]
            labels = labels.to(config["training"]["device"]).long()  # [B, N]

            optimizer.zero_grad()
            outputs = model(points)  # [B, N, num_classes]
            outputs = outputs.reshape(-1, config["training"]["num_classes"])  # (B * N, num_classes)
            labels = labels.view(-1)  # (B * N)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_train_outputs.append(outputs)
            all_train_labels.append(labels)

        # Рассчитываем метрики на тренировочном наборе
        all_train_outputs = torch.cat(all_train_outputs, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)
        train_class_accuracies = calculate_accuracy(
            all_train_outputs, all_train_labels, config["training"]["num_classes"], selected_classes
        )
        print(f"Эпоха {epoch + 1} завершена, Train Loss: {epoch_loss / len(train_loader):.4f}")
        for cls, accuracy in train_class_accuracies.items():
            print(f"Точность для класса {cls} на тренировке: {accuracy:.4f}")

        # Оценка на валидационном наборе
        val_loss, val_class_accuracies = evaluate_model(
            model,
            val_loader,
            criterion,
            config["training"]["num_classes"],
            config["training"]["device"],
            selected_classes=selected_classes
        )

        print(f"Валидация: Средний Loss = {val_loss:.4f}")
        for cls, accuracy in val_class_accuracies.items():
            print(f"Точность для класса {cls} на валидации: {accuracy:.4f}")
