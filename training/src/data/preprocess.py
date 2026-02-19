import os, shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from training.src.config.settings import RAW_DIR, PROCESSED_DIR, IMG_SIZE, BATCH_SIZE, CLASS_DIRS
from common.logger import get_logger

logger = get_logger(__name__)

def preprocess_and_split():
    logger.info("Starting data preprocessing and splitting...")
    classes = CLASS_DIRS
    images, labels = [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(RAW_DIR, cls)
        for fname in os.listdir(cls_dir):
            images.append(os.path.join(cls_dir, fname))
            labels.append(label)

    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.1, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=42
    )

    for split_name, X_split, y_split in zip(
        ["train","val","test"], [X_train,X_val,X_test], [y_train,y_val,y_test]
    ):
        split_dir = os.path.join(PROCESSED_DIR, split_name)
        for cls_name in classes:
            os.makedirs(os.path.join(split_dir, cls_name), exist_ok=True)
        for img_path, label in zip(X_split, y_split):
            shutil.copy(img_path, os.path.join(split_dir, classes[label], os.path.basename(img_path)))
    logger.info("Data preprocessed and split at %s", PROCESSED_DIR)

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

if __name__ == "__main__":
    preprocess_and_split()
