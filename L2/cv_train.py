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

def get_cnn_model(img_size, num_classes):

    inputs = keras.Input(shape=img_size+(3,))
    
    
     # Layer 1: Convolution with ReLU activation
    conv_layer_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv_layer_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_layer_1)
    mp_c1 = MaxPooling2D(pool_size=(2,2))(conv_layer_2)
    
    # Layer 1: Convolution with ReLU activation
    conv_layer_3 = Conv2D(64, (3, 3), padding='same', activation='relu')(mp_c1)
    conv_layer_4 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_layer_3)
    
    fl_1 = Flatten()(conv_layer_4)
    d1 = Dense(64, activation='relu')(fl_1)
    outputs = Dense(num_classes, activation='softmax')(d1)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model
    
    
def get_dense_model(img_size, num_classes):
    model = keras.Sequential() 

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu", input_shape=img_size+(3, ), padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def get_model(base_model, image_size, num_classes):
    # Freeze all the layers
    base_model.trainable = False

    inputs = Input(shape=image_size+(3,))
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model



    
if __name__ == '__main__':
    
    movies_with_img_path = pd.read_csv('./our_data/each_genre200.csv')
    print(movies_with_img_path)
    le = preprocessing.LabelEncoder()
    le.fit(movies_with_img_path['first_genre'])
    movies_with_img_path['genre_encoded'] = le.transform(movies_with_img_path['first_genre'])
    
    X_train, X_val, y_train, y_val = train_test_split(movies_with_img_path.img_path.values, movies_with_img_path.genre_encoded.values, test_size=0.30, random_state=42)
    
    image_size = (200, 150)
    b_size = 128
    train_gen = PosterSequence(b_size, image_size, X_train, y_train)
    val_gen = PosterSequence(b_size, image_size, X_val, y_val)
    
    keras.backend.clear_session()
    
    base_model = tf.keras.applications.InceptionResNetV2(
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
            base_model,  
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(set(movies_with_img_path.genre_encoded.values)), activation='softmax')])
    
    
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
#     base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=image_size+(3, ),
#                                                      include_top=False,
#                                                      weights='imagenet')
    
    # base_model = tf.keras.applications.resnet.ResNet50(input_shape=image_size+(3, ),
    #                                                  include_top=False,
    #                                                  weights='imagenet')


    # model = get_model(base_model=base_model, image_size=image_size, num_classes=len(set(movies_with_img_path.genre_encoded.values)))
    
    # filename='ft_xception_log.csv'
    # history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=2)
    
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
    