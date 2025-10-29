import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

ENV_PATH    = ROOT / "database" / ".env"

load_dotenv(ENV_PATH)

MONGO_CONFIG = {
    'DB_HOST' : os.getenv("MONGO_URL"),
    'DB_NAME' : os.getenv("MONGO_NAME")
}

REMOTE_MONGO_CONFIG = {
    "DB_HOST"   : os.getenv("MONGO_URL_E3A"),
    "DB_NAME"   : os.getenv("MONGO_NAME_E3A")
}
