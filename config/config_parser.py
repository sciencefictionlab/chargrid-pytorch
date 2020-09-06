from dotenv import load_dotenv
from pathlib import Path
import os


def autoconfigure():
    if os.getenv('DIR_NP_CHARGRID_1H') is not None:
        return

    load_dotenv(dotenv_path=Path('.') / '.env', override=True)
    return
