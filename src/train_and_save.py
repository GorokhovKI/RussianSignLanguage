"""
Реализация обучения классификатора жестов на основе извлечённых признаков MediaPipe.
Архитектура:
- предварительный projection (FC) для уменьшения размерности канала
- 1D-CNN по временной оси (Conv1d)
- BiLSTM для моделирования долгосрочных зависимостей
- полносвязный классификатор
Dataset ожидает .npz файлы, содержащие ключ 'landmarks' формы [T, H, 21, 3].
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm
import time
import json
from config import FEATURES_DIR, ANNOTATIONS, MODELS_DIR, METRICS_DIR, PLOTS_DIR, BATCH_SIZE, NUM_EPOCHS, LR, DEVICE, FIXED_LENGTH


class SlovoDataset(Dataset):
    """Dataset читает .npz файлы, возвращает (features, label_idx)

    features: torch.float32 Tensor shape [T, C], где C = H*21*3
    label_idx: int
    """
    def __init__(self, features_dir: Path, annotations_csv: Path, classes: List[str] = None, fixed_length: int = None):
        self.features_dir = Path(features_dir)
        self.ann = pd.read_csv(annotations_csv)
        self.fixed_length = fixed_length
        # сопоставление video_path (из аннотаций) -> npz file
        self.samples = []  # list of tuples (npz_path, label)
        for _, row in self.ann.iterrows():
            video_rel = row['video_path']
            video_stem = Path(video_rel).stem
            npz_path = self.features_dir / f"{video_stem}.npz"
            if npz_path.exists():
                self.samples.append((npz_path, row['label']))
        if classes is None:
            # infer classes from annotations present in samples
            labels = sorted(list({lab for _, lab in self.samples}))
            self.classes = labels
        else:
            self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        d = np.load(path, allow_pickle=True)
        lm = d['landmarks']  # [T, H, 21, 3]
        # flatten per frame
        if lm.ndim == 4:
            T, H, P, K = lm.shape
            feat = lm.reshape(T, H * P * K)
        elif lm.ndim == 3:
            T, P, K = lm.shape
            feat = lm.reshape(T, P * K)
        else:
            raise ValueError(f'Unexpected landmarks shape: {lm.shape}')
        # pad/truncate to fixed_length if requested
        if self.fixed_length is not None:
            feat = self._pad_or_truncate(feat, self.fixed_length)
        # convert to float32
        feat = feat.astype(np.float32)
        label_idx = self.class_to_idx[label]
        return feat, label_idx

    @staticmethod
    def _pad_or_truncate(arr: np.ndarray, target_len: int):
        T = arr.shape[0]
        if T == target_len:
            return arr
        if T > target_len:
            start = max(0, (T - target_len) // 2)
            return arr[start:start+target_len]
        # pad by repeating last frame
        pad_amount = target_len - T
        last = arr[-1:]
        pad = np.repeat(last, pad_amount, axis=0)
        return np.concatenate([arr, pad], axis=0)


def collate_fn(batch):
    feats = [torch.from_numpy(item[0]) for item in batch]  # each [T, C]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    # stack (assume all equal length if fixed_length used)
    X = torch.stack(feats, dim=0)  # [B, T, C]
    return X, labels


class GestureModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, conv_channels: int = 256, lstm_hidden: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=conv_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: [B, T, C]
        x = x.permute(0, 2, 1)  # -> [B, C, T] for Conv1d
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # -> [B, T, conv_channels]
        out, _ = self.lstm(x)  # out [B, T, 2*hidden]
        # take last time step
        out_last = out[:, -1, :]
        logits = self.classifier(out_last)
        return logits


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def train_loop(model, device, train_loader, val_loader, epochs, lr, models_dir, metrics_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    history = []
    early_stop_counter = 0
    patience = 10
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        nb = 0
        for X, y in tqdm(train_loader, desc=f'Epoch {epoch} train'):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            train_acc += accuracy_from_logits(logits, y) * X.size(0)
            nb += X.size(0)
        train_loss /= nb
        train_acc /= nb

        # validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        nbv = 0
        with torch.no_grad():
            for Xv, yv in tqdm(val_loader, desc=f'Epoch {epoch} val'):
                Xv = Xv.to(device)
                yv = yv.to(device)
                logits = model(Xv)
                loss = criterion(logits, yv)
                val_loss += loss.item() * Xv.size(0)
                val_acc += accuracy_from_logits(logits, yv) * Xv.size(0)
                nbv += Xv.size(0)
        val_loss /= nbv
        val_acc /= nbv

        scheduler.step(val_acc)

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}; val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
        # save checkpoint
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, models_dir / f'checkpoint_epoch{epoch}.pth')
        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, models_dir / 'best_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        # early stopping
        if early_stop_counter >= patience:
            print('Early stopping triggered')
            break
    # save history
    pd.DataFrame(history).to_csv(metrics_dir / 'training_history.csv', index=False)
    return history


def main(args):
    features_dir = Path(args.features_dir)
    annotations = Path(args.annotations)
    models_dir = Path(args.models_dir)
    metrics_dir = Path(args.metrics_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # dataset and classes
    dataset = SlovoDataset(features_dir, annotations, fixed_length=args.fixed_length)
    classes = dataset.classes
    num_classes = len(classes)
    print(f'Found {len(dataset)} samples, {num_classes} classes')

    # split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=[dataset.samples[i][1] for i in indices])
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # instantiate model — infer input dim
    sample_feat, _ = dataset[0]
    input_dim = sample_feat.shape[1]
    model = GestureModel(input_dim=input_dim, num_classes=num_classes)
    model.to(DEVICE)

    history = train_loop(model, DEVICE, train_loader, val_loader, epochs=args.epochs, lr=args.lr, models_dir=models_dir, metrics_dir=metrics_dir)
    print('Training finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default=str(FEATURES_DIR))
    parser.add_argument('--annotations', type=str, default=str(ANNOTATIONS))
    parser.add_argument('--models_dir', type=str, default=str(MODELS_DIR))
    parser.add_argument('--metrics_dir', type=str, default=str(METRICS_DIR))
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--fixed_length', type=int, default=FIXED_LENGTH)
    args = parser.parse_args()
    main(args)