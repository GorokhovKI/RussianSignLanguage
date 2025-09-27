"""
plot_metrics.py

Строит графики по training_history.csv и сохраняет их в PLOTS_DIR.
Также предоставляет функцию построения матрицы путаницы по меткам и предсказаниям (если доступны).
"""
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from config import PLOTS_DIR, METRICS_DIR


def plot_history(history_csv: Path, out_dir: Path = PLOTS_DIR):
    df = pd.read_csv(history_csv)
    epochs = df['epoch']
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, df['train_loss'], label='train_loss')
    plt.plot(epochs, df['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs, df['train_acc'], label='train_acc')
    plt.plot(epochs, df['val_acc'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
    out_file = out_dir / 'training_plots.png'
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f'Saved training plots to {out_file}')


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[str], out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=False, fmt='d')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved confusion matrix to {out_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_csv', type=str, default=str(METRICS_DIR / 'training_history.csv'))
    parser.add_argument('--out_dir', type=str, default=str(PLOTS_DIR))
    args = parser.parse_args()
    plot_history(Path(args.history_csv), Path(args.out_dir))