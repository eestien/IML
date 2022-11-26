from tensorflow.keras.preprocessing.image import load_img, array_to_img
from tensorflow import keras
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Flatten, Dense, Activation, Dropout, GlobalAveragePooling2D

from keras import Model, optimizers
from keras.applications import xception
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle


class PosterSequence(keras.utils.Sequence):
    
    def __init__(self, batch_size, img_size, images_paths, labels):
        self.batch_size = batch_size
        self.img_size = img_size
        self.images_paths = images_paths
        self.labels = labels
        self.num_classes = len(set(self.labels))
    
    def __len__(self):
        length = len(self.images_paths) // self.batch_size
        return length
    
    def __getitem__(self, idx):
        x = np.zeros((self.batch_size, ) + self.img_size + (3, ), dtype="float32")
        y = np.zeros((self.batch_size, ) + (self.num_classes, ), dtype="uint8")
        """
        This method returns the batches themselves including images (x) and masks (y) as np.array
            img / 255.0 for images, so that each value is adjacent to the interval [0, 1]
        """
        
        batch_x = self.images_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        x = np.array([keras.preprocessing.image.img_to_array(load_img(file_name, color_mode='rgb', target_size=self.img_size, interpolation='nearest', keep_aspect_ratio=False), dtype="float32") / 255.0 for file_name in batch_x])
        y = np.array([to_categorical(label, num_classes=self.num_classes) for label in batch_y])
        return x, y 
    


    
if __name__ == '__main__':
    np.random.seed(42)
    
    movies_with_img_path = pd.read_csv('./our_data/each_genre200.csv')
    print(movies_with_img_path)
    le = preprocessing.LabelEncoder()
    le.fit(movies_with_img_path['first_genre'])
    movies_with_img_path['genre_encoded'] = le.transform(movies_with_img_path['first_genre'])
    
    # X_train, X_val, y_train, y_val = train_test_split(movies_with_img_path.img_path.values, movies_with_img_path.genre_encoded.values, test_size=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(movies_with_img_path.img_path.values, movies_with_img_path.genre_encoded.values, test_size=0.10, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42, shuffle=True)
    image_size = (200, 150)
    b_size = 32
    train_gen = PosterSequence(b_size, image_size, X_train, y_train)
    val_gen = PosterSequence(b_size, image_size, X_val, y_val)

    keras.backend.clear_session()
    
    base_model = tf.keras.applications.MobileNetV2(
                     include_top=False,
                     weights='imagenet',
                     input_shape=image_size+(3, )
                     )
 
    base_model.trainable=False
    # For freezing the layer we make use of layer.trainable = False
    # means that its internal state will not change during training.
    # model's trainable weights will not be updated during fit(),
    # and also its state updates will not run.
    
    model = tf.keras.Sequential([
            keras.Input(shape=(200, 150, 3)),
            base_model,  
            # Flatten(),
            tf.keras.layers.GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            Dense(64, activation='relu'),
            
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(len(set(movies_with_img_path.genre_encoded.values)), activation='softmax')])
    
    
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")])
    # filename='ft_xception_log.csv'
    # history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
    
    # model.compile(optimizer=keras.optimizers.Adam(lr=1e-2), loss="categorical_crossentropy")
    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-4), metrics=['acc'])


    # Train the model, doing validation at the end of each epoch.
    epochs = 100
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[early_stopping])
    model.save("./ft_xception.h5")
    
    hist_df = pd.DataFrame(history.history) 

    hist_csv_file = 'ft_xception_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    