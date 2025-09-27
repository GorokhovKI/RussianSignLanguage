"""
Цель модуля — надёжно извлечь, нормализовать и сохранить признаки (landmarks MediaPipe) для всего набора Slovo.
Решаемые задачи:
- извлечение landmarks из исходных видео (поддержка 2 рук),
- восполнение пропусков (отсутствующие руки),
- нормализация: root-relative (относительно запястья), масштабирование по динамическому размеру руки,
- упаковка переменной длительности видео: padding/truncation/temporal_resample,
- пакетная обработка с поддержкой многопроцессности для ускорения.

Формат сохранения: для каждого видео создаётся .npz файл с ключами:
- 'landmarks' — float32 массив формы [T, max_hands, 21, 3]
- 'label' — строка
- 'video_id' — строка
- 'fps' — число (если известно)
- 'orig_num_frames' — int

Дополнительно создаётся index.csv с метаданными по каждому сохранённому файлу.
"""
from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('extract_features')

mp_hands = mp.solutions.hands


def _process_frame_landmarks(hand_landmarks_list, max_hands=2):
    """Вспомогательная: преобразовать mediaPipe multi_hand_landmarks в массив [max_hands,21,3]
    hand_landmarks_list — либо None либо iterable объектов LandmarkList
    Порядок рук оставляем таким, каким вернул MediaPipe. Для стабильности можно позже сортировать по x-координате запястья.
    """
    arr = np.zeros((max_hands, 21, 3), dtype=np.float32)
    if not hand_landmarks_list:
        return arr
    for i, hand in enumerate(hand_landmarks_list):
        if i >= max_hands:
            break
        for j, lm in enumerate(hand.landmark):
            arr[i, j, 0] = lm.x
            arr[i, j, 1] = lm.y
            arr[i, j, 2] = lm.z
    return arr


def extract_landmarks_from_video(video_path: Path, max_hands: int = 2, detection_confidence: float = 0.5,
                                 tracking_confidence: float = 0.5) -> dict:
    """Извлекает landmarks и возвращает словарь:
    {
      'landmarks': np.ndarray shape [T, max_hands, 21, 3],
      'fps': float or None,
      'orig_num_frames': int
    }

    Возвращаемые координаты — нормированные относительно размера кадра (как делает MediaPipe).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=max_hands,
                           min_detection_confidence=detection_confidence,
                           min_tracking_confidence=tracking_confidence)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        arr = _process_frame_landmarks(getattr(res, 'multi_hand_landmarks', None), max_hands=max_hands)
        frames.append(arr)
    cap.release()
    hands.close()
    if len(frames) == 0:
        lm = np.zeros((0, max_hands, 21, 3), dtype=np.float32)
    else:
        lm = np.stack(frames, axis=0)
    return {'landmarks': lm, 'fps': fps, 'orig_num_frames': lm.shape[0]}


def normalize_root_relative(landmarks: np.ndarray, root_index: int = 0) -> np.ndarray:
    """Сделать координаты относительными: вычесть координаты root (по умолчанию WRIST — индекс 0 в MediaPipe).
    landmarks: [T, H, 21, 3]
    Возвращаемые значения центрированы, но остаются в системе координат MediaPipe (0..1), поэтому масштабирование ниже также важно.
    """
    if landmarks.size == 0:
        return landmarks
    root = landmarks[:, :, root_index:root_index+1, :2]  # [T,H,1,2]
    out = landmarks.copy()
    out[:, :, :, :2] = out[:, :, :, :2] - root
    return out


def normalize_scale_by_hand(landmarks: np.ndarray, reference_pair=(0, 9)) -> np.ndarray:
    """Масштабировать координаты по интервалу между двумя ключевыми точками (например, WRIST и MIDDLE_FINGER_MCP (index 9) или другой паре).
    Таким образом избавляемся от различий в размере рук и удаляем масштабную компоненту.
    reference_pair: индексы ключевых точек для измерения масштаба
    """
    if landmarks.size == 0:
        return landmarks
    ref_a, ref_b = reference_pair
    # расстояние в каждом кадре и для каждой руки
    a = landmarks[:, :, ref_a, :2]
    b = landmarks[:, :, ref_b, :2]
    d = np.linalg.norm(a - b, axis=-1)  # [T,H]
    # заменить нулевые расстояния на 1.0, чтобы избежать деления на ноль
    d_safe = np.where(d <= 1e-6, 1.0, d)
    out = landmarks.copy()
    out[:, :, :, :2] = out[:, :, :, :2] / d_safe[:, :, None]
    return out


def pad_or_truncate(landmarks: np.ndarray, target_len: int, pad_mode: str = 'edge') -> np.ndarray:
    """Преобразует последовательность кадров к фиксированной длине target_len.
    pad_mode: 'constant' (0), 'edge' (повтор крайних), 'wrap' (циклично)
    """
    T = landmarks.shape[0]
    if T == target_len:
        return landmarks
    if T > target_len:
        # усечём с центровкой
        start = max(0, (T - target_len) // 2)
        return landmarks[start:start+target_len]
    # T < target_len: padding
    pad_amount = target_len - T
    if pad_mode == 'constant':
        pad_arr = np.zeros((pad_amount, *landmarks.shape[1:]), dtype=landmarks.dtype)
        return np.concatenate([landmarks, pad_arr], axis=0)
    elif pad_mode == 'edge':
        last = landmarks[-1:]
        pad_arr = np.repeat(last, pad_amount, axis=0)
        return np.concatenate([landmarks, pad_arr], axis=0)
    elif pad_mode == 'wrap':
        reps = np.ceil(pad_amount / T).astype(int)
        pad_arr = np.tile(landmarks, (reps, 1, 1, 1))[:pad_amount]
        return np.concatenate([landmarks, pad_arr], axis=0)
    else:
        raise ValueError(f"Unknown pad_mode {pad_mode}")


def save_npz(output_path: Path, landmarks: np.ndarray, metadata: dict):
    out = {**metadata, 'landmarks': landmarks}
    # Сохраняем в compressed npz
    np.savez_compressed(output_path, **out)


def process_single_video(args):
    """Вспомогательная функция для многопроцессной обработки: args — tuple с параметрами.
    Возвращает путь файла или ошибку.
    """
    video_path, out_dir, label, video_id, options = args
    try:
        r = extract_landmarks_from_video(video_path, max_hands=options['max_hands'],
                                         detection_confidence=options['detection_confidence'],
                                         tracking_confidence=options['tracking_confidence'])
        lm = r['landmarks']
        orig_frames = r['orig_num_frames']
        if options.get('root_relative', True):
            lm = normalize_root_relative(lm)
        if options.get('scale_normalize', True):
            lm = normalize_scale_by_hand(lm, reference_pair=options.get('scale_reference', (0,9)))
        if options.get('fixed_length', None):
            lm = pad_or_truncate(lm, options['fixed_length'], pad_mode=options.get('pad_mode', 'edge'))
        out_path = out_dir / f"{video_id}.npz"
        metadata = {'label': label, 'video_id': video_id, 'fps': r['fps'], 'orig_num_frames': orig_frames}
        save_npz(out_path, lm, metadata)
        return {'video_id': video_id, 'out_path': str(out_path), 'status': 'ok'}
    except Exception as e:
        return {'video_id': video_id, 'error': str(e), 'status': 'error'}


def process_videos_parallel(videos_root: Path, annotations_csv: Path, out_dir: Path, options: dict, max_workers: int = 4):
    df = pd.read_csv(annotations_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for _, row in df.iterrows():
        video_rel = row['video_path']
        video_path = videos_root / video_rel
        video_id = Path(video_rel).stem
        label = row['label']
        tasks.append((video_path, out_dir, label, video_id, options))
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_single_video, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Processing videos'):
            res = fut.result()
            results.append(res)
    # write index
    idx_rows = [r for r in results if r.get('status') == 'ok']
    pd.DataFrame(idx_rows).to_csv(out_dir / 'index.csv', index=False)
    return results


def convert_mediapipe_json_to_npz(json_path: Path, out_dir: Path, annotations_csv: Path | None = None, options: dict | None = None):
    """Конвертирует slovo_mediapipe.json в .npz файлы с теми же нормализациями, что и при извлечении из видео.
    json формат может быть вариативен — адаптируемся под несколько возможных структур.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = options or {}
    rows = []
    for vid, frames in tqdm(data.items(), desc='Converting mediapipe json'):
        try:
            arr = np.array(frames, dtype=np.float32)
            # возможные форматы: [T, H, 21, 3] или [T, list_of_hands] или [list_of_frames_as_flat]
            if arr.ndim == 1:
                # попытаемся распаковать
                arr = np.stack([np.array(f, dtype=np.float32) for f in frames], axis=0)
            # теперь ожидаем [T,...]
            if arr.ndim == 4:
                lm = arr
            elif arr.ndim == 3:
                # возможно [T, 21, 3] (одна рука) -> добавить размерность рук
                lm = arr[:, None, :, :]
            else:
                # пытаться привести к [T, H, 21, 3]
                lm = arr
            if options.get('root_relative', True):
                lm = normalize_root_relative(lm)
            if options.get('scale_normalize', True):
                lm = normalize_scale_by_hand(lm, reference_pair=options.get('scale_reference', (0,9)))
            if options.get('fixed_length', None):
                lm = pad_or_truncate(lm, options['fixed_length'], pad_mode=options.get('pad_mode','edge'))
            out_path = out_dir / f"{Path(vid).stem}.npz"
            metadata = {'label': None, 'video_id': vid, 'fps': None, 'orig_num_frames': lm.shape[0]}
            save_npz(out_path, lm, metadata)
            rows.append({'video_id': vid, 'out_path': str(out_path), 'status': 'ok'})
        except Exception as e:
            rows.append({'video_id': vid, 'error': str(e), 'status': 'error'})
    pd.DataFrame(rows).to_csv(out_dir / 'index_conversion.csv', index=False)
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract and save MediaPipe landmarks for Slovo dataset')
    parser.add_argument('--videos_root', type=str, help='root folder with video files')
    parser.add_argument('--annotations', type=str, help='path to annotations.csv')
    parser.add_argument('--out_dir', type=str, default='features_npz', help='output directory for .npz files')
    parser.add_argument('--mediapipe_json', type=str, default=None, help='if provided, convert slovo_mediapipe.json instead of processing videos')
    parser.add_argument('--max_workers', type=int, default=4, help='parallel workers')
    parser.add_argument('--fixed_length', type=int, default=None, help='force fixed number of frames per sample')
    parser.add_argument('--pad_mode', type=str, default='edge', choices=['constant','edge','wrap'], help='padding mode when fixed_length is larger than sample length')
    parser.add_argument('--detection_confidence', type=float, default=0.5)
    parser.add_argument('--tracking_confidence', type=float, default=0.5)
    args = parser.parse_args()
    options = {
        'max_hands': 2,
        'detection_confidence': args.detection_confidence,
        'tracking_confidence': args.tracking_confidence,
        'root_relative': True,
        'scale_normalize': True,
        'scale_reference': (0,9),
        'fixed_length': args.fixed_length,
        'pad_mode': args.pad_mode
    }
    out_dir = Path(args.out_dir)
    if args.mediapipe_json:
        convert_mediapipe_json_to_npz(Path(args.mediapipe_json), out_dir, options=options)
    else:
        assert args.videos_root and args.annotations, 'videos_root and annotations are required when not using mediapipe_json'
        process_videos_parallel(Path(args.videos_root), Path(args.annotations), out_dir, options, max_workers=args.max_workers)