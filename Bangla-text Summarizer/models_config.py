# models_config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Your models are exactly here:
# D:\Software\Anaconda\bangla_news_flask\models\saved_model
# D:\Software\Anaconda\bangla_news_flask\models\saved_mt5

DEFAULT_BANGLAT5_PATH = BASE_DIR / "models" / "saved_model"
DEFAULT_MT5_PATH      = BASE_DIR / "models" / "saved_mt5"

BANG_LAT5_PATH = os.getenv("BANG_LAT5_PATH", str(DEFAULT_BANGLAT5_PATH))
MT5_PATH       = os.getenv("MT5_PATH",       str(DEFAULT_MT5_PATH))

print("BanglaT5 model path:", BANG_LAT5_PATH)
print("mT5 model path:", MT5_PATH)
