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
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# Порог для классификации динамического жеста по суммарному перемещению (нормированному)
DEFAULT_MOVEMENT_THRESHOLD = 0.12


def load_annotations(annotations_csv: Path) -> pd.DataFrame:
    """Загружает annotations.csv и возвращает DataFrame с колонками как в датасете.
    Предполагается, что annotations.csv содержит по крайней мере колонки: video_path, label, user_id, start, end
    """
    df = pd.read_csv(annotations_csv, sep='\t')
    return df

def _extract_landmarks_from_frame_repr(frame_repr: Any, max_hands: int = 2) -> np.ndarray:
    """
    Преобразует представление кадра (возможные форматы) в массив shape [H,21,3],
    где H == max_hands. Если руки отсутствуют, возвращается нулевой массив.
    Поддерживаем форматы:
      - list_of_hands: [ hand1, hand2, ... ], где hand = list_of_21_points (каждая point = [x,y,z] или dict {'x':..})
      - frame_dict с ключом 'multi_hand_landmarks' или 'hands' или 'landmarks' и т.п.
      - hand может быть dict с ключами 'landmark' или 'landmarks' (как MediaPipe Python)
      - point может быть dict {'x':..., 'y':..., 'z':...} или список/tuple [x,y,z]
    """
    H = max_hands
    out = np.zeros((H, 21, 3), dtype=np.float32)

    # helper: convert a single point representation to (x,y,z) floats
    def _point_to_xyz(pt):
        if pt is None:
            return (0.0, 0.0, 0.0)
        if isinstance(pt, dict):
            # keys could be 'x','y','z' or similar
            x = float(pt.get('x', pt.get('X', pt.get(0, 0.0))))
            y = float(pt.get('y', pt.get('Y', pt.get(1, 0.0))))
            z = float(pt.get('z', pt.get('Z', pt.get(2, 0.0))))
            return (x, y, z)
        if isinstance(pt, (list, tuple, np.ndarray)):
            # e.g. [x,y,z] or [x,y]
            if len(pt) >= 3:
                return (float(pt[0]), float(pt[1]), float(pt[2]))
            elif len(pt) == 2:
                return (float(pt[0]), float(pt[1]), 0.0)
        # unknown type -> zeros
        return (0.0, 0.0, 0.0)

    # determine representation of the frame
    # Case A: frame_repr is a list -> list of hands or list of landmarks (single hand)
    if isinstance(frame_repr, list):
        # attempt to interpret each element as a hand (list/dict)
        hands = []
        for elem in frame_repr:
            if elem is None:
                continue
            # if elem is a list of 21 points -> treat as hand
            if isinstance(elem, list) and len(elem) in (21,):
                # each point might be [x,y,z] or dict
                hand_points = []
                for pt in elem:
                    hand_points.append(_point_to_xyz(pt))
                hands.append(hand_points)
            elif isinstance(elem, list) and len(elem) > 0 and isinstance(elem[0], (list, tuple, dict)):
                # maybe list of points (len maybe 21) - handle same
                hand_points = []
                for pt in elem:
                    hand_points.append(_point_to_xyz(pt))
                hands.append(hand_points)
            elif isinstance(elem, dict):
                # elem could be a hand-dict with 'landmark' key
                if 'landmark' in elem:
                    pts = elem.get('landmark') or elem.get('landmarks')
                    if isinstance(pts, list):
                        hand_points = [_point_to_xyz(p) for p in pts]
                        hands.append(hand_points)
                else:
                    # fallback: maybe the list is actually list of frames -> not a hand
                    pass
        # if we didn't detect hands above but the list looks like 21 points for single hand
        if len(hands) == 0 and len(frame_repr) == 21:
            # treat frame_repr itself as single hand
            hands.append([_point_to_xyz(pt) for pt in frame_repr])
    elif isinstance(frame_repr, dict):
        # Case B: frame_repr is a dict — try known keys
        # MediaPipe-like: frame_repr could be {'multi_hand_landmarks': [hand1, hand2, ...]}
        candidate_hands = None
        for key in ('multi_hand_landmarks', 'hands', 'landmarks', 'hand_landmarks'):
            if key in frame_repr:
                candidate_hands = frame_repr[key]
                break
        if candidate_hands is None:
            # sometimes the dict is one hand with key 'landmark'
            if 'landmark' in frame_repr:
                candidate_hands = [frame_repr['landmark']]
        if candidate_hands is None:
            # maybe frame_repr itself is mapping of hand-id -> list of points
            # try to find values that are lists of length ~21
            candidate_hands = []
            for v in frame_repr.values():
                if isinstance(v, list) and len(v) in (21,):
                    candidate_hands.append(v)
        hands = []
        if candidate_hands:
            for hand in candidate_hands:
                if isinstance(hand, list):
                    hand_points = [_point_to_xyz(p) for p in hand]
                    hands.append(hand_points)
                elif isinstance(hand, dict) and 'landmark' in hand:
                    pts = hand['landmark']
                    hand_points = [_point_to_xyz(p) for p in pts]
                    hands.append(hand_points)
    else:
        # unsupported type — return zeros
        hands = []

    # now fill into out (pad or truncate to H hands)
    for i, hand_points in enumerate(hands[:H]):
        # ensure hand_points has length 21 (pad with zeros or truncate)
        pts = hand_points
        if len(pts) < 21:
            pts = pts + [(0.0,0.0,0.0)] * (21 - len(pts))
        elif len(pts) > 21:
            pts = pts[:21]
        out[i, :, 0] = [p[0] for p in pts]
        out[i, :, 1] = [p[1] for p in pts]
        out[i, :, 2] = [p[2] for p in pts]
    return out

def load_mediapipe_json(json_path: Path, max_hands: int = 2) -> Dict[str, np.ndarray]:
    """
    Загружает slovo_mediapipe.json и возвращает словарь {video_id: landmarks_array}.
    Поддерживает несколько форматов экспорта:
      - список кадров, каждый кадр — список рук, каждая рука — список точек [x,y,z] или dict {'x','y','z'}
      - список кадров, каждый кадр — dict с ключом 'multi_hand_landmarks'/'hands'/'landmark'
      - и другие близкие форматы
    Финальный формат для каждого видео: numpy array shape [T, max_hands, 21, 3]
    """
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    out = {}
    sample_reported = 0
    for vid, val in raw.items():
        try:
            # val expected to be iterable of frames
            frames = []
            if isinstance(val, dict):
                # sometimes val may be a dict mapping 'frames' or similar
                # try to find list-like member
                possible = None
                for k in ('frames','data','landmarks','annotations','mp_frames'):
                    if k in val and isinstance(val[k], list):
                        possible = val[k]; break
                if possible is not None:
                    iter_frames = possible
                else:
                    # if dict but values are frames, try to sort by key (if keys are numeric)
                    # fallback: treat values() as sequence
                    iter_frames = list(val.values())
            else:
                iter_frames = val

            for fr in iter_frames:
                # each fr can be list/dict/...
                frame_arr = _extract_landmarks_from_frame_repr(fr, max_hands=max_hands)
                frames.append(frame_arr)
            if len(frames) == 0:
                arr = np.zeros((0, max_hands, 21, 3), dtype=np.float32)
            else:
                arr = np.stack(frames, axis=0).astype(np.float32)  # shape [T,H,21,3]
            out[vid] = arr
            # small sample print to help debug structure for first few vids
            if sample_reported < 3:
                print(f"[load_mediapipe_json] video {vid}: parsed frames={arr.shape[0]}, hands={arr.shape[1]}")
                sample_reported += 1
        except Exception as e:
            # catch and explain which video failed
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
    """
    Robust dataset analysis:
    - auto-detects columns for label/video/user if possible (handles your 'attachment_id' + 'text' format),
    - uses precomputed mediapipe dict when available,
    - computes movement and classifies static/dynamic.
    """
    # load annotations (your load_annotations should already use sep='\t' for your file)
    df = load_annotations(annotations_csv)

    # --- infer columns (prefer explicit known names) ---
    cols_lower = {c.lower(): c for c in df.columns.tolist()}
    # possible names for label and video
    if 'text' in cols_lower and 'attachment_id' in cols_lower:
        label_col = cols_lower['text']
        video_col = cols_lower['attachment_id']
        user_col = cols_lower.get('user_id', cols_lower.get('user', ''))
    else:
        # fallback heuristics
        # choose label as a textual column (not UUID-like)
        def looks_like_uuid(s):
            try:
                s = str(s)
                return len(s) >= 36 and s.count('-') >= 4
            except:
                return False

        label_col = None
        for c in df.columns:
            sample_vals = df[c].astype(str).dropna().head(20).tolist()
            # prefer column with some alphabetic characters
            alpha_frac = sum(1 for v in sample_vals if any(ch.isalpha() for ch in v)) / max(1, len(sample_vals))
            uuid_frac = sum(1 for v in sample_vals if looks_like_uuid(v)) / max(1, len(sample_vals))
            if alpha_frac > 0.2 and uuid_frac < 0.5:
                label_col = c
                break
        if label_col is None:
            # fallback to second column if present, else first
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        # video column candidates
        for cand in ('attachment_id', 'video_path', 'video', 'file', 'video_id', 'id'):
            if cand in cols_lower:
                video_col = cols_lower[cand]
                break
        else:
            video_col = df.columns[0] if len(df.columns) > 0 else None

        user_col = cols_lower.get('user_id', cols_lower.get('user', ''))

    print(f"Using columns: label_col='{label_col}', video_col='{video_col}', user_col='{user_col}'")

    # load precomputed mediapipe landmarks if provided
    precomputed = None
    if mediapipe_json and mediapipe_json.exists():
        precomputed = load_mediapipe_json(mediapipe_json)

    results = []
    # ensure results folder exists
    Path('results').mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Analysing'):
        # safely extract label and user_id
        label = ''
        if label_col and label_col in df.columns:
            try:
                label = row[label_col]
                if pd.isna(label):
                    label = ''
                label = str(label).strip()
            except Exception:
                label = ''
        user_id = ''
        if user_col and user_col in df.columns:
            try:
                user_id = row[user_col]
                user_id = '' if pd.isna(user_id) else str(user_id)
            except Exception:
                user_id = ''

        # extract video identifier / filename
        video_field = ''
        if video_col and video_col in df.columns:
            try:
                video_field = row[video_col]
            except Exception:
                video_field = ''
        # normalize video_id: prefer basename without extension (works for UUID or filenames)
        video_id = ''
        if isinstance(video_field, str) and video_field.strip() != '':
            video_id = Path(video_field).name
            # remove extension if present
            video_id = Path(video_id).stem
        else:
            # fallback: try first column value
            try:
                cand = row[df.columns[0]]
                video_id = str(cand) if cand is not None else f"vid_{idx}"
                video_id = Path(video_id).stem
            except Exception:
                video_id = f"vid_{idx}"

        movement = np.nan
        # try several key variants to match precomputed dict keys
        if precomputed:
            found = False
            key_variants = [str(video_id), str(video_id) + '.mp4', str(video_id) + '.avi', str(video_id) + '.mov']
            # also try original value if it contained path
            if isinstance(video_field, str) and video_field.strip():
                key_variants.insert(0, video_field)
                key_variants.insert(0, Path(video_field).name)
                key_variants.insert(0, Path(video_field).stem)
            for k in key_variants:
                if k in precomputed:
                    movement = compute_movement_from_landmarks(np.array(precomputed[k]))
                    found = True
                    break
            if not found:
                # also try raw video_id as-is
                if str(video_id) in precomputed:
                    movement = compute_movement_from_landmarks(np.array(precomputed[str(video_id)]))
        elif videos_root is not None:
            # attempt to extract landmarks on-the-fly
            from extract_features import extract_landmarks_from_video
            candidates = []
            if isinstance(video_field, str) and video_field.strip():
                candidates.append(Path(videos_root) / video_field)
            candidates.append(Path(videos_root) / (str(video_id) + '.mp4'))
            candidates.append(Path(videos_root) / (str(video_id) + '.avi'))
            candidates.append(Path(videos_root) / str(video_id))
            for vp in candidates:
                if vp.exists():
                    lm = extract_landmarks_from_video(vp)
                    movement = compute_movement_from_landmarks(lm)
                    break

        class_type = classify_static_dynamic(movement, movement_threshold) if not np.isnan(movement) else 'unknown'
        results.append({'video_id': str(video_id), 'label': label, 'user_id': user_id, 'movement': movement, 'class_type': class_type})

    out_df = pd.DataFrame(results)
    # save results into results/
    stats_by_label = out_df.groupby(['label', 'class_type']).size().unstack(fill_value=0)
    counts = out_df['label'].value_counts()
    out_df.to_csv('results/analysis_per_video.csv', index=False)
    stats_by_label.to_csv('results/analysis_by_label.csv')
    counts.to_csv('results/counts_by_label.csv')
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