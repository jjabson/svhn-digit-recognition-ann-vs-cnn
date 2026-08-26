from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
GENERATED_DIR = PROJECT_ROOT / "generated"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"

DATA_FILE = DATA_DIR / "SVHN_single_grey1.h5"
MODEL_FILE = MODELS_DIR / "svhn_cnn.keras"

REPORT_SOURCE = REPORTS_DIR / "svhn_report.typ"
REPORT_OUTPUT = REPORTS_DIR / "svhn_report.pdf"
REPORT_DATA_FILE = GENERATED_DIR / "report_data.typ"

DATABASE_FILE = GENERATED_DIR / "ml_framework.db"