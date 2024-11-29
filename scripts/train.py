import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset import create_dataloader
from models.pointnet import PointNet

import os
import json
from utils.dataset import create_dataloader

if __name__ == "__main__":
    # Загружаем конфигурацию
    with open("../config.json", "r") as f:
        config = json.load(f)

    data_dir = config["data"]["output_dir"]
    batch_size = config["training"]["batch_size"]
    num_points = config["data"]["num_points"]
    num_channels = config["data"]["num_channels"]
    split_ratio = config["data"]["train_val_split"]

    # Создаем DataLoader для train и val
    train_loader = create_dataloader(data_dir, batch_size, split="train", split_ratio=split_ratio, num_points=num_points, num_channels=num_channels, shuffle=True)
    val_loader = create_dataloader(data_dir, batch_size, split="val", split_ratio=split_ratio, num_points=num_points, num_channels=num_channels, shuffle=False)

    for points, labels in train_loader:
        print(points.shape, labels.shape)

    model = PointNet(num_classes=config["training"]["num_classes"]).to(config["training"]["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        for points, labels in train_loader:
            points = points.to(config["training"]["device"]).float()  # [B, N, C]
            labels = labels.to(config["training"]["device"]).long()  # [B, N]

            optimizer.zero_grad()
            outputs = model(points)  # [B, N, num_classes]

            # Преобразуем для функции потерь
            outputs = outputs.reshape(-1, config["training"]["num_classes"])  # (B * N, num_classes)
            print("Outputs shape:", outputs.shape)
            labels = labels.view(-1)  # (B * N)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Эпоха {epoch + 1} завершена, Loss: {loss.item()}")
