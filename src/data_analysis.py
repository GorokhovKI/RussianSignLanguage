"""
Научные комментарии:
Данный модуль реализует инструменты для количественного анализа набора данных Slovo.
Основные функции:
- чтение аннотаций (annotations.csv),
- подсчёт примеров на класс,
- определение того, является ли жест динамическим по предвычисленным ключевым точкам или по реальному прогону MediaPipe по видео,
- построение отчётов и вывод статистики в CSV/JSON для последующего использования.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm

# Порог для классификации динамического жеста по суммарному перемещению (нормированному)
DEFAULT_MOVEMENT_THRESHOLD = 0.12


def load_annotations(annotations_csv: Path) -> pd.DataFrame:
    """Загружает annotations.csv и возвращает DataFrame с колонками как в датасете.
    Предполагается, что annotations.csv содержит по крайней мере колонки: video_path, label, user_id, start, end
    """
    df = pd.read_csv(annotations_csv)
    return df


def load_mediapipe_json(json_path: Path) -> Dict[str, np.ndarray]:
    """Загружает slovo_mediapipe.json, возвращает словарь {video_id: landmarks_array}.
    Формат ожидания: для каждого видео массив [num_frames, num_hands, 21, 3] или представление близкое к этому.
    Если формат отличается, функция даёт понятную ошибку.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for vid, val in data.items():
        # предполагаем, что val содержит список кадров, каждый кадр — список рук, каждая рука — список из 21 точек [x,y,z]
        try:
            arr = np.array(val, dtype=np.float32)
            out[vid] = arr
        except Exception as e:
            raise ValueError(f"Не удалось привести landmarks для {vid} к массиву: {e}")
    return out


def compute_movement_from_landmarks(landmarks: np.ndarray) -> float:
    """Оценивает суммарное перемещение ключевых точек по всему видео.
    landmarks: массив формы [num_frames, num_hands, 21, 3] или [num_frames, total_coords]
    Возвращает нормированное суммарное перемещение (float).

    Метод: для каждой руки вычисляется средняя позиция (среднее по 21 точке) в каждом кадре, затем суммируется евклидово расстояние между последовательными кадрами и нормируется на количество кадров.
    """
    if landmarks.ndim == 4:
        # [T, H, 21, 3]
        T = landmarks.shape[0]
        hand_means = landmarks.mean(axis=2)  # [T, H, 3]
        # если рук может быть 0..2, суммируем по существующим
        disp = 0.0
        for h in range(hand_means.shape[1]):
            traj = hand_means[:, h, :2]  # x,y
            # исключаем кадры с нулевыми координатами (отсутствие руки)
            valid_mask = np.any(traj != 0.0, axis=1)
            if valid_mask.sum() < 2:
                continue
            valid_traj = traj[valid_mask]
            diffs = np.sqrt(((valid_traj[1:] - valid_traj[:-1])**2).sum(axis=1))
            disp += diffs.sum() / (valid_traj.shape[0]-1)
        # нормируем по числу рук (1 или 2)
        return disp
    elif landmarks.ndim == 2:
        # [T, coords] ожидаем coords = 21*3*H
        T = landmarks.shape[0]
        coords = landmarks.reshape(T, -1, 3)
        hand_means = coords.mean(axis=1)  # [T, 3]
        traj = hand_means[:, :2]
        valid_mask = np.any(traj != 0.0, axis=1)
        if valid_mask.sum() < 2:
            return 0.0
        valid_traj = traj[valid_mask]
        diffs = np.sqrt(((valid_traj[1:] - valid_traj[:-1])**2).sum(axis=1))
        return diffs.sum() / (valid_traj.shape[0]-1)
    else:
        raise ValueError("Неожиданный размер массива landmarks")


def classify_static_dynamic(movement_value: float, threshold: float = DEFAULT_MOVEMENT_THRESHOLD) -> str:
    """Классификация по порогу: если движение выше threshold => dynamic, иначе static."""
    return 'dynamic' if movement_value > threshold else 'static'


def analyze_dataset(annotations_csv: Path, mediapipe_json: Path | None = None, videos_root: Path | None = None,
                    movement_threshold: float = DEFAULT_MOVEMENT_THRESHOLD) -> pd.DataFrame:
    """Основная функция: возвращает DataFrame с колонками [video_id, label, user_id, movement, class_type]

    Если mediapipe_json предоставлен, использует предвычисленные ключевые точки. Иначе — если videos_root задан, пробует вычислить ключевые точки в режиме on-the-fly (жёсткая операция по времени).
    """
    df = load_annotations(annotations_csv)
    results = []

    precomputed = None
    if mediapipe_json and mediapipe_json.exists():
        precomputed = load_mediapipe_json(mediapipe_json)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Analysing'):
        video_id = str(row.get('video_path') or row.get('video_id') or f"vid_{idx}")
        label = row['label']
        user_id = row.get('user_id', '')
        movement = 0.0
        if precomputed and video_id in precomputed:
            movement = compute_movement_from_landmarks(np.array(precomputed[video_id]))
        elif videos_root is not None:
            from extract_features import extract_landmarks_from_video
            lm = extract_landmarks_from_video(videos_root / video_id)
            movement = compute_movement_from_landmarks(lm)
        else:
            movement = np.nan
        class_type = classify_static_dynamic(movement, movement_threshold) if not np.isnan(movement) else 'unknown'
        results.append({'video_id': video_id, 'label': label, 'user_id': user_id, 'movement': movement, 'class_type': class_type})

    out_df = pd.DataFrame(results)
    # сводная статистика
    stats_by_label = out_df.groupby(['label', 'class_type']).size().unstack(fill_value=0)
    counts = out_df['label'].value_counts()
    # сохраняем отчёты
    out_df.to_csv('analysis_per_video.csv', index=False)
    stats_by_label.to_csv('analysis_by_label.csv')
    counts.to_csv('counts_by_label.csv')
    return out_df


def suggest_augmentation_for_label(count: int, min_target: int = 100) -> List[str]:
    """Если примеров мало, возвращает список рекомендуемых техник аугментации.
    Научное обоснование: для статичных жестов достаточно пространственных преобразований, для динамических — необходимо также варьировать временную компоненту.
    """
    suggestions = []
    if count >= min_target:
        return ['none']
    deficit = min_target - count
    # Базовые трансформации
    suggestions.append('spatial: horizontal_flip, rotation ±10°, scale 0.95-1.05, color jitter')
    # Для динамических жестов полезны временные аугментации
    suggestions.append('temporal: random_crop_duration, time_warp (speed* in [0.9,1.1]), frame_dropout up to 10%')
    # для малых классов добавить oversampling с модификациями
    suggestions.append(f'oversample: generate ~{deficit} samples using combinations of above')
    return suggestions


if __name__ == '__main__':
    # CLI-демонстрация
    import argparse
    parser = argparse.ArgumentParser(description='Analysis tools for Slovo dataset')
    parser.add_argument('--annotations', type=str, required=True, help='path to annotations.csv')
    parser.add_argument('--mediapipe_json', type=str, default=None, help='path to slovo_mediapipe.json')
    parser.add_argument('--videos_root', type=str, default=None, help='root folder with video files')
    parser.add_argument('--threshold', type=float, default=DEFAULT_MOVEMENT_THRESHOLD, help='movement threshold')
    args = parser.parse_args()
    res = analyze_dataset(Path(args.annotations), Path(args.mediapipe_json) if args.mediapipe_json else None,
                          Path(args.videos_root) if args.videos_root else None,
                          movement_threshold=args.threshold)
    print('Saved analysis files: analysis_per_video.csv, analysis_by_label.csv, counts_by_label.csv')