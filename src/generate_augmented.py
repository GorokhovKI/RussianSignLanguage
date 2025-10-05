from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import uuid
from augmentations import random_augment

FEATURES_DIR = Path('results/features')
OUT_DIR = FEATURES_DIR / 'augmented'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_feature_files(features_dir: Path) -> pd.DataFrame:
    files = list(features_dir.glob('*.npz'))
    rows = []
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            label = data.get('label', None)
            vid = data.get('video_id', f.stem)
            rows.append({'path': str(f), 'label': None if label is None else str(label), 'video_id': str(vid)})
        except Exception:
            continue
    return pd.DataFrame(rows)


def generate(features_dir: Path, min_target: int = 100, fixed_length: int = None):
    df = find_feature_files(features_dir)
    df['label'] = df['label'].fillna('UNKNOWN')
    counts = df['label'].value_counts()
    label_to_files = df.groupby('label')['path'].apply(list).to_dict()

    generated = []
    for label, count in counts.items():
        if count >= min_target:
            continue
        deficit = min_target - count
        src_files = label_to_files.get(label, [])
        if len(src_files) == 0:
            continue
        # for simplicity, generate by sampling with replacement from src_files
        i = 0
        while i < deficit:
            src = np.load(src_files[i % len(src_files)], allow_pickle=True)
            lm = src['landmarks']
            # apply augmentation
            aug = random_augment(lm, fixed_length=fixed_length)
            # metadata
            new_id = uuid.uuid4().hex
            out_path = OUT_DIR / f"aug_{label}_{new_id}.npz"
            meta = {'label': label, 'video_id': f'aug_{new_id}', 'fps': src.get('fps', None), 'orig_num_frames': lm.shape[0]}
            np.savez_compressed(out_path, landmarks=aug, **meta)
            generated.append(str(out_path))
            i += 1
    # write index
    pd.DataFrame({'generated_path': generated}).to_csv(features_dir / 'augmented_index.csv', index=False)
    print(f'Generated {len(generated)} augmented samples in {OUT_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default=str(FEATURES_DIR))
    parser.add_argument('--min_target', type=int, default=100)
    parser.add_argument('--fixed_length', type=int, default=None)
    args = parser.parse_args()
    generate(Path(args.features_dir), min_target=args.min_target, fixed_length=args.fixed_length)