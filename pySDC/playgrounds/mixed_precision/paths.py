from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
TABLES_DIR = BASE_DIR / "tables"
PRESENTATION_DIR = IMAGES_DIR / "presentation"


def ensure_output_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)


def image_path(filename):
    ensure_output_dirs()
    return IMAGES_DIR / filename


def table_path(filename):
    ensure_output_dirs()
    return TABLES_DIR / filename


def presentation_path(filename):
    ensure_output_dirs()
    return PRESENTATION_DIR / filename
