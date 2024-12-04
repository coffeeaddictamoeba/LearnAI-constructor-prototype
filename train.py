import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.applications import Xception

class CNN:
    def __init__(self, input_shape, num_classes=2, model=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        if model is not None:
            self.model = tf.keras.models.load_model(model)
        else: self.model = model

    def build_model(self, activation_hidden='relu', activation_output='softmax', dropout=0.5):
        base_model = Xception(input_shape=self.input_shape, weights='imagenet', include_top=False) # pre-built NN architecture
        base_model.trainable = False
        self.model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation=activation_hidden),
            layers.Dropout(dropout),
            layers.Dense(2, activation=activation_output),
        ])
        self.model.summary()

    def reduce_lr(self):
        learning_rate_reduce = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=5,
            verbose=1,
            factor=0.5,
            min_lr=0.00001
        )
        return learning_rate_reduce

    # Configure the learning rate reduction callback
    def set_lr_schedule(self):
        # Create a learning rate schedule using Exponential Decay
        learning_rate_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.5,
        )
        lr_callback = LearningRateScheduler(learning_rate_schedule)
        return lr_callback

    def compile_model(self, set_lr_schedule=False, learning_rate=1e-3):
        if set_lr_schedule:
            learning_rate = self.set_lr_schedule()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            metrics='accuracy'
        )

    def train(self, train_generator, test_generator, filepath, epochs=10, batch_size=32):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call `build_model()` first.")
        early_stopping = EarlyStopping(
            min_delta=0.001,  # minimium amount of change to count as an improvement
            patience=5,  # how many epochs to wait before stopping
            restore_best_weights=True,
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callback = [self.set_lr_schedule(), self.reduce_lr(), early_stopping, checkpoint]
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=test_generator.samples // batch_size,
            callbacks=[callback]
        )
        return history

    def evaluate(self, test_generator):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call `build_model()` first.")
        print("start evaluation")
        print(f"Expected input shape: {self.input_shape}")
        print(f"Actual input shape from generator: {test_generator.image_shape}")
        score = self.model.evaluate(test_generator, verbose=False)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score

    def predict(self, input_data, model=None):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call `build_model()` first.")
        if not model is None:
            self.model = models.load_model(model)
        return self.model.predict(input_data)
