from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List
import os

# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

# Пути и конфигурация
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AugmentedSlovoDataset(Dataset):
    def __init__(self, features_dir: Path, annotations_csv: Path, fixed_length: int = None, augment: bool = False, include_augmented: bool = True):
        self.features_dir = Path(features_dir)
        self.fixed_length = fixed_length
        self.augment = augment
        # build sample list from npz files
        npz_files = list(self.features_dir.glob('*.npz'))
        if include_augmented:
            npz_files += list((self.features_dir / 'augmented').glob('*.npz')) if (self.features_dir / 'augmented').exists() else []
        samples = []
        skipped = 0
        for p in npz_files:
            try:
                data = np.load(p, allow_pickle=True)
                lm = data.get('landmarks', None)
                if lm is None:
                    # если нет landmarks — пропускаем файл
                    skipped += 1
                    continue
                # если нулевая длина (T == 0), пропускаем на этапе инициализации
                if lm.size == 0 or (lm.ndim >= 1 and lm.shape[0] == 0):
                    skipped += 1
                    continue
                label = data.get('label', None)
                if label is None:
                    # try to infer from filename
                    label = p.stem.split('_')[1] if '_' in p.stem else 'UNKNOWN'
                # Привести label к строке, если он не строка
                if not isinstance(label, str):
                    label = str(label)
                # Пропускаем только пустые метки, но оставляем 'no_event'
                if label.lower() in ('', 'none', 'nan'):
                    skipped += 1
                    continue
                samples.append((str(p), str(label)))
            except Exception as e:
                print(f"Error loading {p}: {e}")  # <-- отладка
                skipped += 1
                continue
        print(f"Dataset loaded: {len(samples)} samples, {skipped} skipped")
        self.samples = samples
        if len(samples) == 0:
            raise RuntimeError(f"No valid samples found in {features_dir} (skipped {skipped} files).")
        # infer classes
        labels = sorted(list({lab for _, lab in self.samples}))
        self.classes = labels
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        d = np.load(p, allow_pickle=True)
        lm = d.get('landmarks', None)
        if lm is None:
            # нечего вернуть — безопасно пометить как пустой
            # но так как в __init__ мы отфильтровали такие файлы, это редкость
            if self.fixed_length is None:
                raise ValueError(f"No landmarks in {p} and no fixed_length provided.")
            # create zero landmarks fallback (assume 3D landmarks by default)
            lm = np.zeros((self.fixed_length, 1, 1), dtype=np.float32)

        # если всё же T == 0 (на всякий случай)
        if lm.size == 0 or (lm.ndim >= 1 and lm.shape[0] == 0):
            if self.fixed_length is None:
                raise ValueError(f"Empty landmarks in {p} and no fixed_length specified.")
            # попытаться восстановить форму по оставшимся осям
            if lm.ndim == 3:
                _, P, K = lm.shape
                lm = np.zeros((self.fixed_length, P, K), dtype=np.float32)
            elif lm.ndim == 4:
                _, H, P, K = lm.shape
                lm = np.zeros((self.fixed_length, H, P, K), dtype=np.float32)
            else:
                # универсальный fallback
                lm = np.zeros((self.fixed_length, 1, 1), dtype=np.float32)

        # possibly apply augmentation
        if self.augment:
            # lm = random_augment(lm, fixed_length=self.fixed_length)
            # ^ если у вас есть augmentations, раскомментируйте выше
            pass
            # guard: augmentation должен вернуть хотя бы 1 кадр
            if lm is None or lm.size == 0 or (lm.ndim >= 1 and lm.shape[0] == 0):
                # заменим паддингом
                if self.fixed_length is None:
                    raise ValueError(f"Augmentation produced empty sequence for {p} and fixed_length is None.")
                if lm is None or lm.ndim < 1:
                    lm = np.zeros((self.fixed_length, 1, 1), dtype=np.float32)
                else:
                    # восстановим ожидаемую форму
                    if lm.ndim == 3:
                        _, P, K = lm.shape
                        lm = np.zeros((self.fixed_length, P, K), dtype=np.float32)
                    elif lm.ndim == 4:
                        _, H, P, K = lm.shape
                        lm = np.zeros((self.fixed_length, H, P, K), dtype=np.float32)
                    else:
                        lm = np.zeros((self.fixed_length, 1, 1), dtype=np.float32)
        else:
            if self.fixed_length is not None:
                from extract_features import pad_or_truncate
                lm = pad_or_truncate(lm, self.fixed_length, pad_mode='edge')

        # flatten per frame
        if lm.ndim == 4:
            T, H, P, K = lm.shape
            feat = lm.reshape(T, H * P * K)
        elif lm.ndim == 3:
            T, P, K = lm.shape
            feat = lm.reshape(T, P * K)
        else:
            raise ValueError(f'Unexpected landmarks shape: {lm.shape} from file {p}')
        feat = feat.astype(np.float32)
        label_idx = self.class_to_idx[label]
        return feat, label_idx


def collate_fn(batch):
    # фильтруем случайные пустые примеры, чтобы torch.stack не падал
    feats = []
    labels = []
    for item in batch:
        feat_np, lab = item
        if feat_np is None:
            continue
        if feat_np.shape[0] == 0:
            # пропустить (или можно заменить на нулевой тензор)
            continue
        feats.append(torch.from_numpy(feat_np))
        labels.append(lab)
    if len(feats) == 0:
        raise RuntimeError("All batch samples have zero length (check dataset / augmentation).")
    X = torch.stack(feats, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return X, labels


# Мощная Transformer-модель
class TransformerGestureModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, nhead=8, num_layers=6, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=2048
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # берем последний токен
        return self.fc(x)


def train(args):
    dataset = AugmentedSlovoDataset(
        Path(args.features_dir),
        Path(args.annotations),
        fixed_length=args.fixed_length,
        augment=True,
        include_augmented=args.use_augmented
    )
    labels = [lab for _, lab in dataset.samples]
    idxs = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(idxs, test_size=0.15, random_state=14, stratify=labels, shuffle=True)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(
        AugmentedSlovoDataset(
            Path(args.features_dir),
            Path(args.annotations),
            fixed_length=args.fixed_length,
            augment=False,
            include_augmented=args.use_augmented
        ),
        val_idx
    )

    # DataLoader без WeightedRandomSampler
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    sample_feat, _ = dataset[0]
    input_dim = sample_feat.shape[1]
    model = TransformerGestureModel(
        input_dim=input_dim,
        num_classes=len(dataset.classes),
        nhead=8,
        num_layers=6,
        dropout=0.3
    )
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early Stopping
    best_val_loss = float('inf')
    patience = args.patience
    counter = 0

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.tensorboard_dir)

    history = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        nb = 0
        for X, y in tqdm(train_loader, desc=f'Train epoch {epoch}'):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            running_acc += (logits.argmax(1)==y).float().sum().item()
            nb += X.size(0)
        train_loss = running_loss / nb
        train_acc = running_acc / nb

        # val
        model.eval()
        vloss = 0.0
        vacc = 0.0
        nvt = 0
        with torch.no_grad():
            for Xv, yv in tqdm(val_loader, desc=f'Val epoch {epoch}'):
                Xv = Xv.to(DEVICE)
                yv = yv.to(DEVICE)
                logits = model(Xv)
                loss = criterion(logits, yv)
                vloss += loss.item() * Xv.size(0)
                vacc += (logits.argmax(1)==yv).float().sum().item()
                nvt += Xv.size(0)
        val_loss = vloss / nvt
        val_acc = vacc / nvt

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

        # LR scheduler
        scheduler.step(val_loss)

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'classes': dataset.classes
        }, Path(args.models_dir)/f'checkpoint_epoch{epoch}.pth')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'classes': dataset.classes
            }, Path(args.models_dir)/'best_model.pth')

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    pd.DataFrame(history).to_csv(Path(args.metrics_dir)/'training_history_transformer.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default='results/features')
    parser.add_argument('--annotations', type=str, default='data/annotations.csv')
    parser.add_argument('--models_dir', type=str, default='results_transformer/models')
    parser.add_argument('--metrics_dir', type=str, default='results_transformer/metrics')
    parser.add_argument('--tensorboard_dir', type=str, default='results_transformer/tensorboard')
    parser.add_argument('--batch_size', type=int, default=16)  # уменьшено, т.к. трансформер тяжёлый
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)  # уменьшено
    parser.add_argument('--fixed_length', type=int, default=128)  # увеличили
    parser.add_argument('--use_augmented', type=bool, default=True)
    args = parser.parse_args()

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)

    train(args)