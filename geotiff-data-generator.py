import tensorflow as tf
from PIL import Image
import numpy as np


class CustomDataGenerator(tf.keras.utils.Sequence):

    ''' Custom DataGenerator to load img 
    
    Arguments:
        data_frame = pandas data frame in filenames and labels format
        batch_size = divide data in batches
        shuffle = shuffle data before loading
        img_shape = image shape in (h, w, d) format
        augmentation = data augmentation to make model rebust to overfitting
    
    Output:
        Img: numpy array of image
        label : output label for image
    '''

    def __init__(self, data_frame, batch_size=10, img_shape=None, augmentation=False, num_classes=None):
        self.data_frame = data_frame
        self.train_len = len(data_frame)
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.num_classes = num_classes
        print(
            f"Found {self.data_frame.shape[0]} images belonging to {self.num_classes} classes")

    def __len__(self):
        ''' return total number of batches '''
        self.data_frame = shuffle(self.data_frame)
        return math.ceil(self.train_len/self.batch_size)

    def on_epoch_end(self):
        ''' shuffle data after every epoch '''
        # fix on epoch end it's not working, adding shuffle in len for alternative
        pass

    def __data_augmentation(self, img):
        ''' function for apply some data augmentation '''
        img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.3)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def __get_image(self, file_id):
        """ open image with file_id path and apply data augmentation """
        img = np.asarray(Image.open(file_id))
        img = np.resize(img, self.img_shape)
        # img = self.__data_augmentation(img)
        img = preprocess_input(img)

        return img

    def __get_label(self, label_id):
        """ uncomment the below line to convert label into categorical format """
        #label_id = tf.keras.utils.to_categorical(label_id, num_classes)
        return label_id

    def __getitem__(self, idx):
        batch_x = self.data_frame["filenames"][idx *
                                               self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.data_frame["labels"][idx *
                                            self.batch_size:(idx + 1) * self.batch_size]
        # read your data here using the batch lists, batch_x and batch_y
        x = [self.__get_image(file_id) for file_id in batch_x]
        y = [self.__get_label(label_id) for label_id in batch_y]

        return tf.convert_to_tensor(x), tf.convert_to_tensor(y)
