from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)

IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
EARLY_STOP_PATIENCE = 7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
RANDOM_SEED = 42

COST_FP = 1.0
COST_FN = 5.0

ANOMALY_THRESHOLD_PERCENTILE = 95

CLASSIFIER_PATH = MODEL_DIR / "classifier.pth"
AUTOENCODER_PATH = MODEL_DIR / "autoencoder.pth"
