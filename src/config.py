from pathlib import Path

# Пути к данным (константы проекта)
DATA_ROOT = Path('data')
TRAIN_VIDEOS = DATA_ROOT / 'train'
TEST_VIDEOS = DATA_ROOT / 'test'
ANNOTATIONS = DATA_ROOT / 'annotations.csv'
MP_JSON = DATA_ROOT / 'slovo_mediapipe.json'

# Директория результатов (веса, признаки, графики, метрики)
RESULTS_ROOT = Path('results')
FEATURES_DIR = RESULTS_ROOT / 'features'       # .npz признаки
MODELS_DIR = RESULTS_ROOT / 'models'           # веса и чекпоинты
METRICS_DIR = RESULTS_ROOT / 'metrics'         # csv с метриками
PLOTS_DIR = RESULTS_ROOT / 'plots'             # графики .png
LOGS_DIR = RESULTS_ROOT / 'logs'

# Параметры извлечения признаков
MAX_HANDS = 2
FIXED_LENGTH = 64  # длина временного окна (кадры); можно изменить
PAD_MODE = 'edge'

# Параметры обучения
BATCH_SIZE = 32
NUM_EPOCHS = 60
LR = 1e-3
NUM_CLASSES = None  # будет заполнено при чтении annotations
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

# Сопутствующие
DEFAULT_MOVEMENT_THRESHOLD = 0.12

# Создание директорий при импорте (без перезаписи)
for p in [RESULTS_ROOT, FEATURES_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)