import os
from pathlib import Path
from common.base import *

# Paths - workspace root is 3 levels up from this file
BASE_DIR = str(Path(__file__).resolve().parents[3])

# Temp file directory for uploaded images
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp")

# UVicorn server settings
HOST_NAME = "0.0.0.0"
PORT = 8000

# Comments