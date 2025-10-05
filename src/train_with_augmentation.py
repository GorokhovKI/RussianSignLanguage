from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List
import random
from config import DEVICE, FIXED_LENGTH
from augmentations import random_augment


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
                samples.append((str(p), str(label)))
            except Exception:
                # можно логгировать имя файла при желании
                skipped += 1
                continue
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
            lm = random_augment(lm, fixed_length=self.fixed_length)
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


# Model — reuse GestureModel from previous file or define inline (simple reproducible network)
class GestureModelSmall(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        o, _ = self.lstm(x)
        o = o[:, -1, :]
        return self.fc(o)


def compute_class_weights(dataset: AugmentedSlovoDataset):
    counts = {}
    for _, lab in dataset.samples:
        counts[lab] = counts.get(lab, 0) + 1
    labels = list(dataset.class_to_idx.keys())
    class_counts = np.array([counts.get(l, 0) for l in labels], dtype=np.float32)
    # inverse frequency
    weights = 1.0 / (class_counts + 1e-6)
    # normalize
    weights = weights / weights.sum() * len(labels)
    return torch.tensor(weights, dtype=torch.float)


def train(args):
    dataset = AugmentedSlovoDataset(Path(args.features_dir), Path(args.annotations), fixed_length=args.fixed_length, augment=True, include_augmented=args.use_augmented)
    labels = [lab for _, lab in dataset.samples]
    idxs = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(idxs, test_size=0.15, random_state=14, stratify=labels, shuffle=True)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(AugmentedSlovoDataset(Path(args.features_dir), Path(args.annotations), fixed_length=args.fixed_length, augment=False, include_augmented=args.use_augmented), val_idx)

    train_files = {dataset.samples[i][0] for i in train_idx}
    val_files = {dataset.samples[i][0] for i in val_idx}
    print("Train size:", len(train_files))
    print("Val size:", len(val_files))
    print("Intersection size:", len(train_files & val_files))
    if len(train_files & val_files) > 0:
        print("Примеры пересекаются! Примеры:", list(train_files & val_files)[:10])

    from collections import Counter
    train_labels = [dataset.samples[i][1] for i in train_idx]
    val_labels = [dataset.samples[i][1] for i in val_idx]
    print("Train:", len(train_labels), " Val:", len(val_labels))
    print("Train label counts (top 20):", Counter(train_labels).most_common(20))
    print("Val label counts (top 20):", Counter(val_labels).most_common(20))

    # weighted sampler for train
    class_weights = compute_class_weights(dataset)
    sample_weights = [class_weights[dataset.class_to_idx[dataset.samples[i][1]]].item() for i in train_idx]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    sample_feat, _ = dataset[0]
    input_dim = sample_feat.shape[1]
    model = GestureModelSmall(input_dim=input_dim, num_classes=len(dataset.classes))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        nb = 0
        for X,y in tqdm(train_loader, desc=f'Train epoch {epoch}'):
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

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
        # save checkpoint
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': dataset.classes}, Path(args.models_dir)/f'checkpoint_epoch{epoch}.pth')
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'classes': dataset.classes}, Path(args.models_dir)/'best_model.pth')

    pd.DataFrame(history).to_csv(Path(args.metrics_dir)/'training_history_aug.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default='results/features')
    parser.add_argument('--annotations', type=str, default='data/annotations.csv')
    parser.add_argument('--models_dir', type=str, default='results/models')
    parser.add_argument('--metrics_dir', type=str, default='results/metrics')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fixed_length', type=int, default=64)
    parser.add_argument('--use_augmented', type=bool, default=True)
    args = parser.parse_args()
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)
    train(args)