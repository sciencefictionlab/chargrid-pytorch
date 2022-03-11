from pathlib import Path
import os
from config import autoconfigure

autoconfigure()

dir_dicts = {
    "DIR_NP_CHARGRID_1H": os.getenv("DIR_NP_CHARGRID_1H"),
    "DIR_NP_GT_1H": os.getenv("DIR_NP_GT_1H"),
    "DIR_NP_BBOX_ANCHOR_MASK": os.getenv("DIR_NP_BBOX_ANCHOR_MASK"),
    "DIR_NP_BBOX_ANCHOR_COORD": os.getenv("DIR_NP_BBOX_ANCHOR_COORD"),
    "DIR_IMG": os.getenv("DIR_IMG"),
    "DIR_BOXES": os.getenv("DIR_BOXES"),
    "DIR_CLASSES": os.getenv("DIR_CLASSES"),
    "DIR_NP_CHARGRID": os.getenv("DIR_NP_CHARGRID"),
    "DIR_PNG_CHARGRID": os.getenv("DIR_PNG_CHARGRID"),
    "DIR_NP_GT": os.getenv("DIR_NP_GT"),
    "DIR_PNG_GT": os.getenv("DIR_PNG_GT"),
    "DIR_PD_BBOX": os.getenv("DIR_PD_BBOX"),
    "DIR_NP_CHARGRID_REDUCED": os.getenv("DIR_NP_CHARGRID_REDUCED"),
    "DIR_PNG_CHARGRID_REDUCED": os.getenv("DIR_PNG_CHARGRID_REDUCED"),
    "DIR_NP_GT_REDUCED": os.getenv("DIR_NP_GT_REDUCED"),
    "DIR_PNG_GT_REDUCED": os.getenv("DIR_PNG_GT_REDUCED"),
    "DIR_PD_BBOX_REDUCED": os.getenv("DIR_PD_BBOX_REDUCED"),
}


def check_dataset_dir_present():
    for key in dir_dicts:
        if not Path(dir_dicts[key]).exists():
            os.mkdir(dir_dicts[key])
        else:
            print(Path(dir_dicts[key]))
            print("it exists")


if __name__ == "__main__":
    check_dataset_dir_present()
