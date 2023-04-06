import tensorflow as tf
from tensorflow.keras.preprocessing import sequence




class preproc():
    
    """
    Description
    
    -------
    params:
    
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    augmentation - ...
    tokenize - ...
    
    """


    def __init__(self, config, augmentation, tokenize):
        super().__init__()
        self.config = config
        self.augmentation = augmentation
        self.tokenize = tokenize


    def process_image(self, image):
        return self.augmentation(image = image.numpy())['image']

    def tokenize_(self, descriptions):
        return sequence.pad_sequences(self.tokenize.texts_to_sequences(descriptions), 
                                      maxlen = self.config.MAX_SEQUENCE_LENGTH)

    def tokenize_text(self, text):
        return self.tokenize_([text.numpy().decode('utf-8')])[0]

    def tf_process_train_dataset_element(self, image, table_data, text, price):
        im_shape = image.shape
        [image,] = tf.py_function(self.process_image, [image], [tf.uint8])
        image.set_shape(im_shape)
        [text,] = tf.py_function(self.tokenize_text, [text], [tf.int32])
        return (image, table_data, text), price

    def tf_process_val_dataset_element(self, image, table_data, text, price):
        [text,] = tf.py_function(self.tokenize_text, [text], [tf.int32])
        return (image, table_data, text), price