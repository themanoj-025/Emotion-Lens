import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import kagglehub

def parse_args():
    parser = argparse.ArgumentParser(description="Train Face Emotion Detection CNN Model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--model_name', type=str, default='emotion_model.h5', help='Name of the saved model file (default: emotion_model.h5)')
    return parser.parse_args()

def main():
    args = parse_args()

    print("Downloading FER2013 dataset from Kaggle...")
    path = kagglehub.dataset_download("msambare/fer2013")
    print("Path to dataset files:", path)

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")

    print("Training Data Path:", train_path)
    print("Testing Data Path:", test_path)

    img_width, img_height = 48, 48
    batch_size = args.batch_size

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    print(f"\nLoading Training Data (Batch Size: {batch_size})...")
    try:
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical'
        )
    except Exception as e:
        print("Could not load training data. Check path.", e)
        train_generator = None

    print("\nLoading Testing/Validation Data...")
    try:
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical'
        )
    except Exception as e:
        print("Could not load testing data. Check path.", e)
        test_generator = None

    if not train_generator or not test_generator:
        print("Data loading failed. Exiting.")
        return

    print("\nBuilding the CNN Model...")

    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second Convolutional Block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third Convolutional Block
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flattening to convert 2D features to 1D vector
    model.add(Flatten())

    # Fully Connected (Dense) Layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # Output Softmax Layer (7 emotions)
    num_classes = 7
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    print(f"\nStarting Training for {args.epochs} Epochs...")

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=args.epochs,
        validation_data=test_generator,
        validation_steps=test_generator.n // test_generator.batch_size
    )

    model.save(args.model_name)
    print(f"\nModel saved successfully as '{args.model_name}'")

if __name__ == "__main__":
    main()
