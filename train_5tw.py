import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Conv2D, Conv3D, Concatenate, MaxPooling2D, UpSampling2D
import keras 
from tensorflow import keras
import tensorflow as tf


from tqdm import tqdm
from datetime import date, timedelta

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date) / np.timedelta64(1, 'h'))):
        yield start_date + timedelta(hours=n)

def generate_image_set(data, start_date=pd.Timestamp('2020-01-25T23'), end_date=pd.Timestamp('2020-01-25T23'), window_size=5):
    train_3d_images = []
    ground_truth = []
    gt_dict = {}
    # end_date = data.timestamp.max()
    
    end_date = end_date-timedelta(hours=window_size-1)
    
    for single_date in tqdm(daterange(start_date, end_date), total=len(list(daterange(start_date, end_date)))):
        three_d_image = []
        inner_end_date = single_date + timedelta(hours=window_size)
        ground_truth.append(generate_image(spec_timestamp=inner_end_date, input_data=data))
        gt_dict[inner_end_date] = ground_truth[-1]
        for inner_single_data in daterange(single_date, inner_end_date):
            three_d_image.append(generate_image(spec_timestamp=inner_single_data, input_data=data))
        train_3d_images.append(three_d_image)
    return train_3d_images, ground_truth, single_date, gt_dict


def generate_image(spec_timestamp, input_data):
    data = np.zeros((round((LON_MAX_BOUND-LON_MIN_BOUND) / 0.0005)+1, round((LAT_MAX_BOUND-LAT_MIN_BOUND) / 0.0005)+1), dtype=np.uint8)
    for _, row in input_data[(input_data.timestamp==spec_timestamp)][['lat', 'lon', 'num_of_posts']].iterrows():
        lon, lat, n_posts = row.lon, row.lat, int(row.num_of_posts)
        try:
            data[round((LON_MAX_BOUND-lon) / 0.0005), round((LAT_MAX_BOUND-lat) / 0.0005)] = n_posts * 100
        except Exception as e:
            print(e)
            print(lon, lat)
            print(round((LON_MAX_BOUND-lon) / 0.0005), round((LAT_MAX_BOUND-lat) / 0.0005))
    return data

class ImgSequence(keras.utils.Sequence):
    
    def __init__(self, batch_size, img_size, images_paths, target_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.images_paths = images_paths
        self.target_paths = target_paths
        
    
    def __len__(self):
        length = 0
        # Each item in the sequence has a size of batch_size, this method returns the number of batches. Shoud be integer!
        # TODO: length calculation
        length = len(self.images_paths) // self.batch_size
        return length
    
    def __getitem__(self, idx):
        # x = np.zeros((self.batch_size, ) + self.img_size + (3, ), dtype="float32")
        # y = np.zeros((self.batch_size, ) + self.img_size + (1, ), dtype="uint8")
        """
        This method returns the batches themselves including images (x) and masks (y) as np.arrays.
        Correspondingly x contains the batch_size of the pet images, y the true segmentation of the images from x.
        Note:

                img / 255.0 for images, so that each value is adjacent to the interval [0, 1] 
                
        """
        
        # TODO x and y calculation
        batch_x = self.images_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.target_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        x = np.array([np.array(file_name).reshape((787, 422, -1, 1)) / float(pixels_norm * 100) for file_name in batch_x])
        y = np.array([np.expand_dims(np.array(file_name), axis=(2, 3)) / float(pixels_norm * 100) for file_name in batch_y])
        return x, y 
        

# def get_unet_model(img_size):

#     inputs = keras.Input(shape=img_size+(5, 1))
    
#     # --- Encoder ---
#     # first conv layer
#     conv_layer_1_1 = Conv3D(10, (1, 1, 3), activation=None, padding='valid')(inputs)
#     act1= tf.keras.activations.elu(conv_layer_1_1, alpha=0.0)
#     conv_layer_1_2 = Conv3D(20, (1, 1, 3), activation=None, padding='valid')(act1)
#     act2= tf.keras.activations.elu(conv_layer_1_2, alpha=0.0)
#     conv_layer_1_3 = Conv3D(40, (3, 3, 1), activation='sigmoid', padding='same')(act2)
#     act3= tf.keras.activations.elu(conv_layer_1_3, alpha=0.0)
    
#     conv_layer_2_1 = Conv3D(20, (3, 3, 1), activation='sigmoid', padding='same')(act3)
#     act4= tf.keras.activations.elu(conv_layer_2_1, alpha=0.0)
#     conv_layer_2_2 = Conv3D(10, (3, 3, 1), activation='sigmoid', padding='same')(act4)
#     act5= tf.keras.activations.elu(conv_layer_2_2, alpha=0.0)
#     outputs = Conv3D(1, 1, activation='sigmoid', padding='same')(act5)
    
#     model = keras.Model(inputs, outputs)
#     return model


def get_unet_model(img_size):

    inputs = keras.Input(shape=img_size+(5, 1))
    
    # --- Encoder ---
    # first conv layer
    inputs_pad = tf.keras.layers.ZeroPadding3D(padding=(1, 1, 0), data_format=None)(inputs)
    print(inputs_pad.shape)
    conv_layer_1_1 = Conv3D(10, (3, 3, 3), activation='sigmoid', padding='valid')(inputs_pad)
    conv_layer_1_1 = tf.keras.layers.ZeroPadding3D(padding=(1, 1, 0), data_format=None)(conv_layer_1_1)
    conv_layer_1_2 = Conv3D(20, (3, 3, 3), activation='sigmoid', padding='valid')(conv_layer_1_1)
    conv_layer_1_3 = Conv3D(40, (3, 3, 1), activation='sigmoid', padding='same')(conv_layer_1_2)
    
    conv_layer_2_1 = Conv3D(20, (3, 3, 1), activation='sigmoid', padding='same')(conv_layer_1_3)
    conv_layer_2_2 = Conv3D(10, (3, 3, 1), activation='sigmoid', padding='same')(conv_layer_2_1)
    outputs = Conv3D(1, 1, activation='sigmoid', padding='same')(conv_layer_2_2)
    
    model = keras.Model(inputs, outputs)
    return model


# train_X_0, train_X_1, train_X_2, train_X_3 = np.load('./train_X_0.npy'), np.load('./train_X_1.npy'), np.load('./train_X_2.npy'), np.load('./train_X_3.npy')
# train_y_0, train_y_1, train_y_2, train_y_3 = np.load('./train_y_0.npy'), np.load('./train_y_1.npy'), np.load('./train_y_2.npy'), np.load('./train_y_3.npy')
if __name__ == '__main__':
    valid_no_outliers = pd.read_pickle('./valid_processed.pickle')
    LAT_MIN_BOUND, LAT_MAX_BOUND = valid_no_outliers.lat.min(), valid_no_outliers.lat.max()
    LON_MIN_BOUND, LON_MAX_BOUND = valid_no_outliers.lon.min(), valid_no_outliers.lon.max()
    pixels_norm = 540
    train_X_0, train_X_1, train_X_2, train_X_3 = np.load('./train_X_0.npy'), np.load('./train_X_1.npy'), np.load('./train_X_2.npy'), np.load('./train_X_3.npy')
    train_y_0, train_y_1, train_y_2, train_y_3 = np.load('./train_y_0.npy'), np.load('./train_y_1.npy'), np.load('./train_y_2.npy'), np.load('./train_y_3.npy')
    
    valid_X, valid_y, last_date, v_gt = generate_image_set(data=valid_no_outliers, start_date=pd.Timestamp('2020-02-01T09'), end_date=pd.Timestamp('2020-02-29T22'))


    print('Numpy arrays uploaded!')
    train_X = list(train_X_0) +list(train_X_1)+list(train_X_2)+list(train_X_3)
    train_y = list(train_y_0) +list(train_y_1)+list(train_y_2)+list(train_y_3)
    # train_X = list(np.load('./tw_7_train_data/trainX.npy'))
    # train_y = list(np.load('./tw_7_train_data/trainy.npy'))
    print("Numpy arrays merged!")
    # Instantiate data Sequences for each split
    batch_size = 64
    train_gen = ImgSequence(batch_size=batch_size, img_size=(787, 422), images_paths=train_X, target_paths=train_y)
    
    valid_gen = ImgSequence(batch_size=batch_size, img_size=(787, 422), images_paths=valid_X, target_paths=valid_y)

    
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    model = get_unet_model((787, 422))

    model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.5, name="Adadelta"), loss=tf.keras.losses.MeanAbsolutePercentageError())
    
    filename='log_tw5_sigmoid.csv'
    history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

    # Train the model, doing validation at the end of each epoch.
    epochs = 25
    model.fit(train_gen, validation_data=valid_gen, epochs=epochs, callbacks=[history_logger]) # validation_data=valid_gen
    print('Model is saving...')
    model.save("./model_tw5_sigmoid.h5")

