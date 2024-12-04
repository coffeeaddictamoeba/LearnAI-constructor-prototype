import os
import random
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

class LearnAIDataset:
    def __init__(self, train_path, test_path, image_size, num_classes, dataset_name=None, link_to_orig=None):
        self.train_path = train_path
        self.test_path = test_path
        self.image_size = image_size
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.link_to_orig = link_to_orig

    def save_dataset(self):
        # TODO: add logic for "Add my own dataset button" (not for MVP)
        pass

    def get_categories(self):
        categories = set()
        if not os.path.exists(self.train_path):
            raise ValueError(f"Dataset path '{self.train_path}' does not exist.")

        # List all files in the dataset path
        for file_name in os.listdir(self.train_path):
            categories.add(file_name)

        if not categories:
            raise ValueError(f"No valid categories found in dataset path '{self.train_path}'.")

        return list(categories)

    def get_sample_images(self, num_samples=5, resize_to=None):
        """
        Get a balanced set of sample images from the dataset with the directory structure:
        train_path/
            category1/
                image1.jpg
                image2.jpg
            category2/
                image1.jpg
                image2.jpg
        Optionally resize images to the specified dimensions.

        Args:
            num_samples (int): The total number of sample images to return.
            resize_to (tuple): Optional. A tuple specifying the target size (width, height) for resizing images.

        Returns:
            List[Tuple[str, Image.Image]]: List of tuples containing file paths and resized image objects.
        """
        categories = self.get_categories()
        sample_images = {category: [] for category in categories}

        # Group images by category
        for category in categories:
            category_path = os.path.join(self.train_path, category)
            if os.path.isdir(category_path):
                for file_name in os.listdir(category_path):
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(category_path, file_name)
                        sample_images[category].append(file_path)

        final_samples = []
        base_samples_per_category = num_samples // len(categories)  # Base samples for each category
        extra_samples = num_samples % len(categories)  # Handle the remainder

        for i, (category, files) in enumerate(sample_images.items()):
            num_to_sample = base_samples_per_category
            if i < extra_samples:
                num_to_sample += 1

            # Ensure we don't sample more than available
            if len(files) > 0:
                final_samples.extend(random.sample(files, min(len(files), num_to_sample)))

        # Optionally resize images
        if resize_to:
            resized_samples = []
            for file_path in final_samples:
                try:
                    img = Image.open(file_path)
                    if resize_to:
                        img = img.resize(resize_to)
                    resized_samples.append((file_path, img))  # Store path and resized image
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")
            final_samples = resized_samples

        return final_samples

    def preprocess_image(self, image_path, image_size):
        target_size = image_size if image_size else self.image_size
        image = load_img(image_path, target_size=target_size)
        return np.expand_dims(image, axis=0)  # Add batch dimension

    def augment_data_for_training(self, batch_size=32):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,  # Rescale pixel values to [0, 1]
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        batch_size = batch_size
        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.image_size,
            batch_size=batch_size,
            seed=32,
            shuffle=True,
            class_mode='categorical'
        )
        print("train data successfully augmented")
        return train_generator

    def augment_data_for_testing(self, batch_size=32):
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Rescale pixel values to [0, 1]
        test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical')
        print("test data successfully augmented")
        return test_generator

    def create_batch(self, batch_size=32, desired_count=1000):
        train_generator = self.augment_data_for_training(batch_size)
        subset_images = []
        subset_labels = []

        # Loop through the generator until you have collected the desired number of images
        for i in range(desired_count):
            # Generate a batch of data (images and labels)
            batch = train_generator.next()
            images, labels = batch
            subset_images.extend(images)
            subset_labels.extend(labels)

            # Check if you've collected enough images
            if len(subset_images) >= desired_count:
                break

        X_sub = np.array(subset_images)
        y_sub = np.array(subset_labels)
        return X_sub, y_sub

