"""
Robust feature extraction and conversion utilities for Slovo dataset.

Features:
- extract MediaPipe hand landmarks from videos (up to 2 hands)
- convert Mediapipe JSON dumps to per-video .npz files
- normalize root-relative coordinates and scale by hand size
- pad / truncate to fixed temporal length
- skip and report videos with zero frames or missing labels
- runtime-friendly: supports multi-process processing

Saved .npz format (only for valid samples):
- 'landmarks' : float32 array shape [T, max_hands, 21, 3]
- 'label'     : str (always a string)
- 'video_id'  : str
- 'fps'       : float or None
- 'orig_num_frames' : int

This file replaces the project's original extract_features.py with more defensive logic
to avoid producing empty .npz files and to ensure labels are propagated correctly.
"""
from __future__ import annotations
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm


logger = logging.getLogger("extract_features")
logging.basicConfig(level=logging.INFO)

mp_hands = mp.solutions.hands


# ---------------------- low-level helpers ----------------------

def _point_to_xyz(pt: Any) -> Tuple[float, float, float]:
    """Convert a point representation to a tuple (x,y,z).

    Supports dict {'x','y','z'}, list/tuple/ndarray [x,y,z], None.
    """
    if pt is None:
        return 0.0, 0.0, 0.0
    if isinstance(pt, dict):
        # tolerant access
        x = pt.get('x', pt.get('X', None))
        y = pt.get('y', pt.get('Y', None))
        z = pt.get('z', pt.get('Z', 0.0))
        # fallback to values order
        try:
            if x is None or y is None:
                vals = list(pt.values())
                if len(vals) >= 2:
                    x = x if x is not None else vals[0]
                    y = y if y is not None else vals[1]
                if len(vals) >= 3:
                    z = z if (z is not None) else vals[2]
        except Exception:
            pass
        try:
            return float(x), float(y), float(z)
        except Exception:
            return 0.0, 0.0, 0.0
    if isinstance(pt, (list, tuple, np.ndarray)):
        if len(pt) >= 3:
            return float(pt[0]), float(pt[1]), float(pt[2])
        if len(pt) == 2:
            return float(pt[0]), float(pt[1]), 0.0
    try:
        f = float(pt)
        return f, 0.0, 0.0
    except Exception:
        return 0.0, 0.0, 0.0


def _process_frame_landmarks(hand_landmarks_list: Optional[Iterable[Any]], max_hands: int = 2) -> np.ndarray:
    """Convert MediaPipe results (or similar) to array [H,21,3]."""
    arr = np.zeros((max_hands, 21, 3), dtype=np.float32)
    if not hand_landmarks_list:
        return arr
    for i, hand in enumerate(hand_landmarks_list):
        if i >= max_hands:
            break
        # mediaPipe LandmarkList has attribute landmark
        pts = getattr(hand, 'landmark', None)
        if pts is None:
            # maybe hand already a list/dict
            try:
                # try iterate
                for j, lm in enumerate(hand):
                    if j >= 21:
                        break
                    x, y, z = _point_to_xyz(lm)
                    arr[i, j, 0] = x
                    arr[i, j, 1] = y
                    arr[i, j, 2] = z
            except Exception:
                continue
        else:
            for j, lm in enumerate(pts):
                if j >= 21:
                    break
                arr[i, j, 0] = lm.x
                arr[i, j, 1] = lm.y
                arr[i, j, 2] = lm.z
    return arr


# ---------------------- extraction pipeline ----------------------

def extract_landmarks_from_video(video_path: Path, max_hands: int = 2, detection_confidence: float = 0.5,
                                 tracking_confidence: float = 0.5) -> Dict[str, Any]:
    """Run MediaPipe Hands over the video and return landmarks and metadata.

    Returns dict with keys: 'landmarks' (np.ndarray [T,H,21,3]), 'fps', 'orig_num_frames'
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=max_hands,
                           min_detection_confidence=detection_confidence,
                           min_tracking_confidence=tracking_confidence)

    frames: List[np.ndarray] = []
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
        lm = np.stack(frames, axis=0).astype(np.float32)
    return {'landmarks': lm, 'fps': fps, 'orig_num_frames': int(lm.shape[0])}


def normalize_root_relative(landmarks: np.ndarray, root_index: int = 0) -> np.ndarray:
    if landmarks.size == 0:
        return landmarks
    root = landmarks[:, :, root_index:root_index+1, :2]  # [T,H,1,2]
    out = landmarks.copy()
    out[:, :, :, :2] = out[:, :, :, :2] - root
    return out


def normalize_scale_by_hand(landmarks: np.ndarray, reference_pair: Tuple[int, int] = (0, 9)) -> np.ndarray:
    if landmarks.size == 0:
        return landmarks
    ref_a, ref_b = reference_pair
    a = landmarks[:, :, ref_a, :2]
    b = landmarks[:, :, ref_b, :2]
    d = np.linalg.norm(a - b, axis=-1)  # [T,H]
    d_safe = np.where(d <= 1e-6, 1.0, d)
    out = landmarks.copy()
    out[:, :, :, :2] = out[:, :, :, :2] / d_safe[:, :, None]
    return out


def pad_or_truncate(landmarks: np.ndarray, target_len: int, pad_mode: str = 'edge') -> np.ndarray:
    T = landmarks.shape[0]
    if T == target_len:
        return landmarks
    if T > target_len:
        start = max(0, (T - target_len) // 2)
        return landmarks[start:start+target_len]
    # T < target_len: pad
    pad_amount = target_len - T
    if pad_mode == 'constant':
        pad_arr = np.zeros((pad_amount, *landmarks.shape[1:]), dtype=landmarks.dtype)
        return np.concatenate([landmarks, pad_arr], axis=0)
    elif pad_mode == 'edge':
        if T == 0:
            # nothing to repeat -> return zeros
            return np.zeros((target_len, *landmarks.shape[1:]), dtype=landmarks.dtype)
        last = landmarks[-1:]
        pad_arr = np.repeat(last, pad_amount, axis=0)
        return np.concatenate([landmarks, pad_arr], axis=0)
    elif pad_mode == 'wrap':
        if T == 0:
            return np.zeros((target_len, *landmarks.shape[1:]), dtype=landmarks.dtype)
        reps = int(np.ceil(pad_amount / T))
        pad_arr = np.tile(landmarks, (reps, 1, 1, 1))[:pad_amount]
        return np.concatenate([landmarks, pad_arr], axis=0)
    else:
        raise ValueError(f"Unknown pad_mode {pad_mode}")


def save_npz(output_path: Path, landmarks: np.ndarray, metadata: Dict[str, Any]) -> None:
    # ensure label is a string (or omit saving if label is None)
    meta = metadata.copy()
    if 'label' in meta and meta['label'] is not None:
        meta['label'] = str(meta['label'])
    else:
        meta['label'] = ''
    out = {**meta, 'landmarks': landmarks}
    np.savez_compressed(output_path, **out)


# ---------------------- single-video processor ----------------------

def process_single_video(args: Tuple[Path, Path, Any, str, Dict[str, Any]]) -> Dict[str, Any]:
    video_path, out_dir, label, video_id, options = args
    try:
        r = extract_landmarks_from_video(video_path, max_hands=options['max_hands'],
                                         detection_confidence=options['detection_confidence'],
                                         tracking_confidence=options['tracking_confidence'])
        lm = r['landmarks']
        orig_frames = int(r['orig_num_frames'])

        # skip empty sequences
        if orig_frames == 0 or lm.size == 0 or (lm.ndim >= 1 and lm.shape[0] == 0):
            return {'video_id': video_id, 'out_path': None, 'status': 'empty', 'reason': 'no_frames'}

        if options.get('root_relative', True):
            lm = normalize_root_relative(lm)
        if options.get('scale_normalize', True):
            lm = normalize_scale_by_hand(lm, reference_pair=options.get('scale_reference', (0, 9)))
        if options.get('fixed_length', None):
            lm = pad_or_truncate(lm, options['fixed_length'], pad_mode=options.get('pad_mode', 'edge'))

        # label must be present as a meaningful string
        meta_label = None if label is None else str(label).strip()
        if meta_label is None or meta_label == '' or meta_label.lower() in ('none', 'nan'):
            return {'video_id': video_id, 'out_path': None, 'status': 'empty', 'reason': 'no_label'}

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_id}.npz"
        metadata = {'label': meta_label, 'video_id': video_id, 'fps': r.get('fps', None), 'orig_num_frames': orig_frames}
        save_npz(out_path, lm, metadata)
        return {'video_id': video_id, 'out_path': str(out_path), 'status': 'ok'}
    except Exception as e:
        logger.exception("Error processing video %s", video_id)
        return {'video_id': video_id, 'error': str(e), 'status': 'error'}


# ---------------------- batch processing ----------------------

def process_videos_parallel(videos_root: Path, annotations_csv: Path, out_dir: Path, options: Dict[str, Any], max_workers: int = 4) -> List[Dict[str, Any]]:
    df = pd.read_csv(annotations_csv, sep='\t')
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []

    # annotations CSV may have different column names. Try common variants.
    if 'video_path' in df.columns and 'label' in df.columns:
        vids = df['video_path'].tolist()
        labels = df['label'].tolist()
    elif 'file' in df.columns and 'label' in df.columns:
        vids = df['file'].tolist()
        labels = df['label'].tolist()
    elif 'video_id' in df.columns and 'label' in df.columns:
        vids = df['video_id'].tolist()
        labels = df['label'].tolist()
    else:
        raise ValueError('annotations_csv must contain columns (video_path|file|video_id) and label')

    for video_rel, label in zip(vids, labels):
        video_path = Path(videos_root) / str(video_rel)
        video_id = Path(video_rel).stem
        tasks.append((video_path, out_dir, label, video_id, options))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_single_video, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Processing videos'):
            res = fut.result()
            results.append(res)

    # write index
    pd.DataFrame(results).to_csv(out_dir / 'index.csv', index=False)
    return results


# ---------------------- convert mediapipe json -> npz ----------------------

def convert_mediapipe_json_to_npz(json_path: Path, out_dir: Path, annotations_csv: Optional[Path] = None, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    # build label map from annotations csv if provided
    label_map: Dict[str, str] = {}
    if annotations_csv is not None and Path(annotations_csv).exists():
        try:
            df_ann = pd.read_csv(annotations_csv, sep='\t', header=None)
            if df_ann.shape[1] >= 2:
                for _, row in df_ann.iterrows():
                    vid = str(row[0]).strip()
                    lbl = str(row[1]).strip()
                    if lbl and lbl.lower() not in ('', 'none', 'nan', 'no_event'):
                        label_map[vid] = lbl
        except Exception:
            logger.exception("Failed to parse annotations CSV for label mapping")

    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    options = options or {}
    results: List[Dict[str, Any]] = []
    total = len(raw)

    for vid, frames in tqdm(raw.items(), desc='Converting mediapipe json', total=total):
        try:
            # frames may be many formats; try to extract per-frame representations
            if isinstance(frames, dict):
                candidate = None
                for k in ('frames', 'data', 'landmarks', 'mp_frames', 'annotations'):
                    if k in frames and isinstance(frames[k], list):
                        candidate = frames[k]
                        break
                if candidate is not None:
                    iter_frames = candidate
                else:
                    iter_frames = list(frames.values())
            else:
                iter_frames = frames

            parsed_frames: List[np.ndarray] = []
            for fr in iter_frames:
                parsed = _extract_landmarks_from_frame_repr(fr, max_hands=options.get('max_hands', 2))
                parsed_frames.append(parsed)

            if len(parsed_frames) == 0:
                arr = np.zeros((0, options.get('max_hands', 2), 21, 3), dtype=np.float32)
            else:
                arr = np.stack(parsed_frames, axis=0).astype(np.float32)

            # optional normalizations
            if options.get('root_relative', True) and arr.shape[0] > 0:
                root = arr[:, :, 0:1, :2]
                arr[:, :, :, :2] = arr[:, :, :, :2] - root
            if options.get('scale_normalize', True) and arr.shape[0] > 0:
                a = arr[:, :, 0, :2]
                b = arr[:, :, 9, :2]
                d = np.linalg.norm(a - b, axis=-1)
                d_safe = np.where(d <= 1e-6, 1.0, d)[:, :, None, None]
                arr[:, :, :, :2] = arr[:, :, :, :2] / d_safe
            if options.get('fixed_length', None):
                arr = pad_or_truncate(arr, options['fixed_length'], pad_mode=options.get('pad_mode', 'edge'))

            # label from label_map if available
            video_id = str(Path(vid).stem)
            label = label_map.get(video_id, None)

            # skip saving if no frames or no label
            if arr.shape[0] == 0:
                results.append({'video_id': vid, 'out_path': None, 'status': 'empty', 'reason': 'no_frames'})
                continue
            if label is None or str(label).strip().lower() in ('', 'none', 'nan'):
                results.append({'video_id': vid, 'out_path': None, 'status': 'empty', 'reason': 'no_label'})
                continue

            out_path = out_dir / f"{video_id}.npz"
            meta = {'label': str(label), 'video_id': vid, 'fps': None, 'orig_num_frames': int(arr.shape[0])}
            save_npz(out_path, arr, meta)
            results.append({'video_id': vid, 'out_path': str(out_path), 'status': 'ok'})
        except Exception as e:
            logger.exception("Error converting mediapipe json entry %s", vid)
            results.append({'video_id': vid, 'error': str(e), 'status': 'error'})

    pd.DataFrame(results).to_csv(out_dir / 'index_conversion.csv', index=False)
    return results


# ---------------------- legacy robust frame repr parser ----------------------

def _extract_landmarks_from_frame_repr(frame_repr: Any, max_hands: int = 2) -> np.ndarray:
    """Robust parser for many possible frame representations; returns [H,21,3]"""
    H = max_hands
    out = np.zeros((H, 21, 3), dtype=np.float32)
    hands: List[List[Tuple[float, float, float]]] = []

    # list-like top-level
    if isinstance(frame_repr, list):
        for elem in frame_repr:
            if elem is None:
                continue
            if isinstance(elem, dict) and ('landmark' in elem or 'landmarks' in elem):
                pts = elem.get('landmark') or elem.get('landmarks')
                if isinstance(pts, list):
                    hands.append([_point_to_xyz(p) for p in pts])
            elif isinstance(elem, list) and (len(elem) == 21 or (len(elem) > 0 and isinstance(elem[0], (list, dict)))):
                hands.append([_point_to_xyz(p) for p in elem])
            else:
                # unknown - skip
                pass
        if len(hands) == 0 and len(frame_repr) == 21:
            hands.append([_point_to_xyz(p) for p in frame_repr])

    # dict-like top-level
    elif isinstance(frame_repr, dict):
        for key in ('multi_hand_landmarks', 'hands', 'landmarks', 'hand_landmarks'):
            if key in frame_repr:
                candidate = frame_repr[key]
                if isinstance(candidate, list):
                    for hand in candidate:
                        if isinstance(hand, dict) and ('landmark' in hand or 'landmarks' in hand):
                            pts = hand.get('landmark') or hand.get('landmarks')
                            if isinstance(pts, list):
                                hands.append([_point_to_xyz(p) for p in pts])
                        elif isinstance(hand, list):
                            hands.append([_point_to_xyz(p) for p in hand])
                    break
        if len(hands) == 0 and 'landmark' in frame_repr:
            pts = frame_repr.get('landmark')
            if isinstance(pts, list):
                hands.append([_point_to_xyz(p) for p in pts])
        if len(hands) == 0:
            for v in frame_repr.values():
                if isinstance(v, list) and len(v) in (21,):
                    hands.append([_point_to_xyz(p) for p in v])
    else:
        # unknown type -> keep empty
        hands = []

    for i, hp in enumerate(hands[:H]):
        pts = hp
        if len(pts) < 21:
            pts = pts + [(0.0, 0.0, 0.0)] * (21 - len(pts))
        elif len(pts) > 21:
            pts = pts[:21]
        out[i, :, 0] = [p[0] for p in pts]
        out[i, :, 1] = [p[1] for p in pts]
        out[i, :, 2] = [p[2] for p in pts]
    return out


# ---------------------- CLI ----------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract MediaPipe landmarks or convert mediapipe json to per-video .npz')
    parser.add_argument('--videos_root', type=str, help='root folder with video files')
    parser.add_argument('--annotations', type=str, help='path to annotations.csv (required for videos mode)')
    parser.add_argument('--out_dir', type=str, default='results/features', help='output directory for .npz files')
    parser.add_argument('--mediapipe_json', type=str, default=None, help='if provided, convert mediapipe json instead of processing videos')
    parser.add_argument('--max_workers', type=int, default=4, help='parallel workers')
    parser.add_argument('--fixed_length', type=int, default=None, help='force fixed number of frames per sample')
    parser.add_argument('--pad_mode', type=str, default='edge', choices=['constant', 'edge', 'wrap'])
    parser.add_argument('--detection_confidence', type=float, default=0.5)
    parser.add_argument('--tracking_confidence', type=float, default=0.5)
    args = parser.parse_args()

    options = {
        'max_hands': 2,
        'detection_confidence': args.detection_confidence,
        'tracking_confidence': args.tracking_confidence,
        'root_relative': True,
        'scale_normalize': True,
        'scale_reference': (0, 9),
        'fixed_length': args.fixed_length,
        'pad_mode': args.pad_mode,
    }

    out_dir = Path(args.out_dir)

    if args.mediapipe_json:
        results = convert_mediapipe_json_to_npz(Path(args.mediapipe_json), out_dir, annotations_csv=Path(args.annotations) if args.annotations else None, options=options)
        ok = [r for r in results if r.get('status') == 'ok']
        empty = [r for r in results if r.get('status') == 'empty']
        err = [r for r in results if r.get('status') == 'error']
        logger.info('Conversion finished. ok=%d, empty=%d, error=%d', len(ok), len(empty), len(err))
    else:
        assert args.videos_root and args.annotations, 'videos_root and annotations are required when not using mediapipe_json'
        results = process_videos_parallel(Path(args.videos_root), Path(args.annotations), out_dir, options, max_workers=args.max_workers)
        ok = [r for r in results if r.get('status') == 'ok']
        empty = [r for r in results if r.get('status') == 'empty']
        err = [r for r in results if r.get('status') == 'error']
        logger.info('Extraction finished. ok=%d, empty=%d, error=%d', len(ok), len(empty), len(err))