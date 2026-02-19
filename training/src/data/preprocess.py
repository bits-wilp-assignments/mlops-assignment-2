import os
import argparse
import warnings
from PIL import ImageFile, Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from training.src.config.settings import RAW_DIR, PROCESSED_DIR, IMG_SIZE, BATCH_SIZE, CLASS_DIRS
from common.logger import get_logger

logger = get_logger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress PIL warnings during tests
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

def preprocess_and_split():
    logger.info("Starting data preprocessing and splitting (with Resizing)...")
    classes = CLASS_DIRS
    images, labels = [], []

    # 1. Gather files and filter out hidden files (like .DS_Store)
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(RAW_DIR, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(cls_dir, fname))
                labels.append(label)

    # 2. Split 80/10/10
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.1, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=42
    )

    # 3. Process and Save
    for split_name, X_split, y_split in zip(
        ["train", "val", "test"], [X_train, X_val, X_test], [y_train, y_val, y_test]
    ):
        split_dir = os.path.join(PROCESSED_DIR, split_name)
        logger.info(f"Processing {split_name} split...")

        for img_path, label in zip(X_split, y_split):
            try:
                # PHYSICAL PREPROCESSING:
                # load_img automatically resizes and filters out non-image files
                img = load_img(img_path, target_size=IMG_SIZE)

                # Create class directory if not exists
                target_cls_dir = os.path.join(split_dir, classes[label])
                os.makedirs(target_cls_dir, exist_ok=True)

                # Save as a new file (this changes the hash for DVC)
                save_path = os.path.join(target_cls_dir, os.path.basename(img_path))
                img.save(save_path)

            except Exception as e:
                # This catches the "Truncated File" and other corruptions
                logger.warning(f"Skipping corrupt image {img_path}: {e}")
                continue

    logger.info("Data resized and split successfully at %s", PROCESSED_DIR)

def get_generators():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    train_gen = datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )
    val_gen = datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(PROCESSED_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    return train_gen, val_gen, test_gen

def main():
    parser = argparse.ArgumentParser(description='Preprocess and split image data for training')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run data preprocessing and splitting')
    args = parser.parse_args()

    if args.preprocess:
        preprocess_and_split()
    else:
        # Default behavior: run preprocessing
        preprocess_and_split()

if __name__ == "__main__":
    main()
