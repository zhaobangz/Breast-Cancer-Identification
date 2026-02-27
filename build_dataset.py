"""Build dataset splits using the centralized data pipeline.

This script uses `cancernet.data_pipeline` to discover, validate,
and stratify-split an image dataset and then copies files into
`config.TRAIN_PATH`, `config.VAL_PATH`, and `config.TEST_PATH`.

Run:
        python build_dataset.py

No external keys or network access required.
"""

from cancernet import config
from cancernet.data_pipeline import create_splits
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
        logging.info("Starting dataset build using cancernet.data_pipeline")
        create_splits(
                input_dir=config.INPUT_DATASET,
                output_base=config.BASE_PATH,
                train_split=config.TRAIN_SPLIT,
                val_split=config.VAL_SPLIT,
                seed=7,
        )
        logging.info("Dataset build finished. Check %s", config.BASE_PATH)
