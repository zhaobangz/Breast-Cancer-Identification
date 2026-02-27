"""Data pipeline helpers for discovering, validating and splitting images.

This module intentionally avoids any network or external-key usage.
All operations are local file I/O and PIL validation. Designed for
reproducible, stratified splits for classification datasets.
"""

import os
import shutil
import random
from imutils import paths
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import List, Tuple


def discover_images(input_dir: str) -> List[str]:
    """Return a list of image file paths under `input_dir`."""
    return list(paths.list_images(input_dir))


def get_label_from_path(p: str) -> str:
    """Infer label from parent directory if possible, else fallback to
    the filename parsing used in the original repository.
    """
    parent = os.path.basename(os.path.dirname(p))
    if parent.isdigit():
        return parent
    # fallback: original project encoded label in filename (last 5th char)
    file = os.path.basename(p)
    if len(file) >= 5 and file[-5:-4].isdigit():
        return file[-5:-4]
    return "0"


def validate_image(p: str) -> bool:
    """Quick validate that PIL can open the image. Return False if broken."""
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def stratified_split(paths_list: List[str], labels: List[str], train_split: float, val_split: float, seed: int = 7) -> Tuple[List[str], List[str], List[str]]:
    """Stratified split into train/val/test.

    val_split is fraction of original data (e.g., 0.1 for 10% validation).
    """
    test_size = 1.0 - train_split
    # first split train vs temp (val+test)
    train_paths, temp_paths, train_lbls, temp_lbls = train_test_split(
        paths_list, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    # compute validation fraction relative to temp
    if test_size == 0:
        return train_paths, [], []

    val_fraction_of_temp = val_split / test_size
    val_paths, test_paths, _, _ = train_test_split(
        temp_paths, temp_lbls, test_size=(1.0 - val_fraction_of_temp), stratify=temp_lbls, random_state=seed
    )

    return train_paths, val_paths, test_paths


def create_splits(input_dir: str, output_base: str, train_split: float = 0.8, val_split: float = 0.1, seed: int = 7) -> None:
    """Discover images, validate them, create stratified splits and copy
    images into the output directory structure.

    Output layout:
      <output_base>/training/<label>/*.jpg
      <output_base>/validation/<label>/*.jpg
      <output_base>/testing/<label>/*.jpg
    """
    all_paths = discover_images(input_dir)
    random.seed(seed)

    # validate and filter
    valid_paths = []
    labels = []
    for p in all_paths:
        if validate_image(p):
            lbl = get_label_from_path(p)
            valid_paths.append(p)
            labels.append(lbl)

    if len(valid_paths) == 0:
        raise RuntimeError(f"No valid images found in {input_dir}")

    train_paths, val_paths, test_paths = stratified_split(valid_paths, labels, train_split, val_split, seed)

    datasets = [("training", train_paths), ("validation", val_paths), ("testing", test_paths)]

    for set_name, paths_list in datasets:
        base_path = os.path.sep.join([output_base, set_name])
        os.makedirs(base_path, exist_ok=True)
        for p in paths_list:
            lbl = get_label_from_path(p)
            label_dir = os.path.sep.join([base_path, lbl])
            os.makedirs(label_dir, exist_ok=True)
            dst = os.path.sep.join([label_dir, os.path.basename(p)])
            # copy2 preserves metadata; keep local-only, no external network access
            shutil.copy2(p, dst)
