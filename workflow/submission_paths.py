"""Runtime paths for curated scientific sources (no study inputs are bundled).

The historical helper name is retained to minimize changes to the algorithms.
All generated artifacts live below an ignored work directory, not beside code.
"""

import os
from pathlib import Path


def code():
    return Path(__file__).resolve().parent


def root():
    return Path(os.environ.get('PROJECT_WORK_DIR', code().parent / 'runs/workflow')).expanduser().resolve()


def source():
    return Path(os.environ.get('PROJECT_SOURCE_DIR', root() / 'input')).expanduser().resolve()


def dataset():
    return Path(os.environ.get('PROJECT_DATASET_DIR', root() / 'datasets/rcc2025necrosis')).expanduser().resolve()


def tokens():
    return Path(os.environ.get('PROJECT_TOKEN_DIR', root() / 'tile_tokens')).expanduser().resolve()


def images():
    return Path(os.environ.get('PROJECT_IMAGES_DIR', source() / 'IMC_images')).expanduser().resolve()
