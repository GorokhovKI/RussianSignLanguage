from __future__ import annotations
import numpy as np
from typing import Tuple
import random
from scipy import interpolate

from extract_features import pad_or_truncate


def horizontal_flip(landmarks: np.ndarray) -> np.ndarray:
    """Зеркальное отображение по горизонтали (x -> -x)."""
    if landmarks.size == 0:
        return landmarks
    out = landmarks.copy()
    out[:, :, :, 0] = -out[:, :, :, 0]
    return out


def rotate_xy(landmarks: np.ndarray, angle_deg: float) -> np.ndarray:
    """Поворот координат x,y вокруг origin (0,0) на angle_deg градусов.
    angle_deg положителен против часовой стрелки.
    """
    if landmarks.size == 0:
        return landmarks
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    out = landmarks.copy()
    xy = out[:, :, :, :2]
    # применим матричное вращение к каждому вектору
    x = xy[..., 0]
    y = xy[..., 1]
    xr = c * x - s * y
    yr = s * x + c * y
    out[:, :, :, 0] = xr
    out[:, :, :, 1] = yr
    return out


def scale(landmarks: np.ndarray, factor: float) -> np.ndarray:
    if landmarks.size == 0:
        return landmarks
    out = landmarks.copy()
    out[:, :, :, :2] = out[:, :, :, :2] * factor
    out[:, :, :, 2] = out[:, :, :, 2] * factor
    return out


def translate(landmarks: np.ndarray, tx: float, ty: float) -> np.ndarray:
    out = landmarks.copy()
    out[:, :, :, 0] = out[:, :, :, 0] + tx
    out[:, :, :, 1] = out[:, :, :, 1] + ty
    return out


def gaussian_jitter(landmarks: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    out = landmarks.copy()
    noise = np.random.normal(scale=sigma, size=out[..., :2].shape)
    out[..., :2] = out[..., :2] + noise
    return out


# --- temporal ---

def time_warp(landmarks: np.ndarray, speed_factor: float) -> np.ndarray:
    """Изменение скорости: интерполируем временную ось.
    speed_factor >1 -> ускорить (меньше кадров), <1 -> замедлить (больше кадров).
    """
    if landmarks.size == 0:
        return landmarks
    T = landmarks.shape[0]
    new_T = max(1, int(np.round(T / speed_factor)))
    # интерполяция по временной оси для каждого координатного канала
    coords = landmarks.reshape(T, -1)  # [T, C]
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_T)
    f = interpolate.interp1d(x_old, coords, axis=0, kind='linear', fill_value='extrapolate')
    new_coords = f(x_new)
    new_landmarks = new_coords.reshape(new_T, *landmarks.shape[1:])
    return new_landmarks.astype(np.float32)


def frame_dropout(landmarks: np.ndarray, drop_prob: float = 0.1) -> np.ndarray:
    if landmarks.size == 0:
        return landmarks
    T = landmarks.shape[0]
    keep_mask = np.random.rand(T) > drop_prob
    if keep_mask.sum() == 0:
        # не удаляем все
        keep_mask[random.randint(0, T-1)] = True
    new = landmarks[keep_mask]
    return new


def random_crop(landmarks: np.ndarray, target_len: int) -> np.ndarray:
    T = landmarks.shape[0]
    if T <= target_len:
        return landmarks
    start = random.randint(0, T - target_len)
    return landmarks[start:start+target_len]


def temporal_shift(landmarks: np.ndarray, shift: int) -> np.ndarray:
    """Shift frames left/back, pad with edge frames."""
    if landmarks.size == 0:
        return landmarks
    T = landmarks.shape[0]
    out = np.zeros_like(landmarks)
    if shift >= 0:
        out[shift:] = landmarks[:T-shift]
        out[:shift] = np.repeat(landmarks[:1], shift, axis=0)
    else:
        s = -shift
        out[:T-s] = landmarks[s:]
        out[T-s:] = np.repeat(landmarks[-1:], s, axis=0)
    return out


# Helper: compose random transforms

def random_augment(landmarks: np.ndarray, fixed_length: int = None) -> np.ndarray:
    """Применить случайную комбинацию пространственных и временных трансформаций.
    fixed_length: если задано, гарантируем длину результата (усечь/дополнить повтором края).
    """
    out = landmarks.copy()
    # spatial probabilistic
    if random.random() < 0.5:
        out = horizontal_flip(out)
    if random.random() < 0.5:
        angle = random.uniform(-12, 12)
        out = rotate_xy(out, angle)
    if random.random() < 0.4:
        fac = random.uniform(0.95, 1.05)
        out = scale(out, fac)
    if random.random() < 0.4:
        tx = random.uniform(-0.03, 0.03)
        ty = random.uniform(-0.03, 0.03)
        out = translate(out, tx, ty)
    if random.random() < 0.4:
        out = gaussian_jitter(out, sigma=0.01)

    # temporal probabilistic
    if random.random() < 0.5:
        speed = random.uniform(0.9, 1.1)
        out = time_warp(out, speed)
    if random.random() < 0.3:
        out = frame_dropout(out, drop_prob=0.08)
    # ensure length
    if fixed_length is not None:
        out = pad_or_truncate(out, fixed_length, pad_mode='edge')
    return out