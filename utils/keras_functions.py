import numpy as np
import pandas as pd
import os
import re

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer




def clean_stopwords(str_2_clean):

    """
    Function for clearing input string from numbers, signs, repeated characters
   
    return: cleared input string
    -------
    params:
    str_2_clean - string to be creared from numbers, signs, repeated characters
    
    """
    
    stopwords_list = stopwords.words('english') + stopwords.words('russian')
    morph = MorphAnalyzer()

    str_2_clean=re.sub(r'[^\w\s]|([0-9])', ' ', str_2_clean)
    str_2_clean=re.sub(' +', ' ', str_2_clean.lower()).strip(' ')
    str_2_clean=re.sub("(.)\\1{2,}", "\\1", str_2_clean)
    words = str_2_clean.split() 
    clean_words = [morph.normal_forms(word)[0] for word in words 
                   if (word not in stopwords_list) and len(word)>2]
    return " ".join(clean_words)



def filtetred_freq_words(max_freq_to_remove, min_freq_to_remove, dict_from_data):
    
    """
    Function makes the dictionary with words that occur too often and too rarely
   
    return: filtered_dict - dictionary with too often and too rarely words
    -------
    params:
    max_freq_to_remove - an integer specifying the number of rare words
    min_freq_to_remove - an integer specifying the number of often words
    dict_from_data - the dict of tokenize.word_index defines a number 
        that determines how many times a word has occurred in the data 
    
    """
    
    filtered_dict = dict(
        filter(lambda item: (item[1]<max_freq_to_remove)or(item[1]>min_freq_to_remove), 
               dict_from_data.items()))
    return filtered_dict


def get_clean_description(data_descr, config):

    """
    Function for clearing feature 'description' of concatenate df_train and df_test 
        from numbers, signs, repeated characters
   
    return: data_descr - cleared feature 'description' of concatenate df_train and df_test 
        from numbers, signs, repeated characters
    -------
    params:
    data_descr - feature 'description' of concatenate df_train and df_test
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    
    """

    # Cleaning from stopwords
    data_descr = data_descr.apply(clean_stopwords)
    print('clean stopwords of data.description done')
    
    tokenize = Tokenizer(num_words=config.MAX_WORDS)
    tokenize.fit_on_texts(data_descr)
    
    ## the words that we will filter
    filtered_values = set(filtetred_freq_words(100, 5000, tokenize.word_index).keys()) #(100, 9000)
    
    ## exclude filtered words
    data_descr = data_descr.apply(
        lambda x: " ".join([word for word in x.split() if word not in filtered_values]))

    return data_descr



def data_descr_to_nlp(data_descr, config, list_inds):   

    """
    Function ...
   
    return: text_train_sequences, text_test_sequences, text_sub_sequences - 
        ...  
    tokenize - ... 
    -------
    params:
    data_descr - feature 'description' of concatenate df_train and df_test 
        cleared from rare and often occuring words 
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    list_inds - list of data indexes X_sub, X_train, X_test
    
    """
    
    ## Tokenize the cleaned and transformed description data description
    tokenize = Tokenizer(num_words=config.MAX_WORDS)
    tokenize.fit_on_texts(data_descr)
    
    ## data split
    text_train = data_descr.iloc[list_inds[0]]
    text_test = data_descr.iloc[list_inds[1]]
    text_sub = data_descr.iloc[list_inds[2]]
    
    ## ...
    text_train_sequences = sequence.pad_sequences(
        tokenize.texts_to_sequences(text_train), maxlen=config.MAX_SEQUENCE_LENGTH)
    text_test_sequences = sequence.pad_sequences(
        tokenize.texts_to_sequences(text_test), maxlen=config.MAX_SEQUENCE_LENGTH)
    text_sub_sequences = sequence.pad_sequences(
        tokenize.texts_to_sequences(text_sub), maxlen=config.MAX_SEQUENCE_LENGTH)
    
    return text_train_sequences, text_test_sequences, text_sub_sequences, tokenize



# Callbacks that used for training model
def callbacks(PATH_BEST_MODEL, config):                               

    """
    Function creates callback list from Keras callbacks, that used in fit method
    List consists 3 function: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    (for a detailed description, see: https://keras.io/api/callbacks/)
    
    -------
    params:

    PATH_BEST_MODEL - path to save model with best params during model train
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)

    """

    checkpoint = ModelCheckpoint(os.path.join(PATH_BEST_MODEL, config.best_weights_name), #PATH_BEST_MODEL,   
                                 monitor=config.monitor_m_checkpnt, 
                                 verbose=config.verbose_m_checkpnt, 
                                 mode=config.mode_m_checkpnt, 
                                 save_best_only=config.save_best_only
                                 )

    earlystop = EarlyStopping(monitor=config.monitor_early_stop, 
                              patience=config.patience_early_stop, 
                              restore_best_weights=config.restore_best_weights
                              )

    reduce_lr = ReduceLROnPlateau(monitor=config.monitor_reduce_plteau, 
                                  factor=config.factor_reduce_plteau, 
                                  patience=config.patience_reduce_plteau, 
                                  verbose=config.verbose_reduce_plteau,
                                  min_lr=config.LR/config.min_lr_ratio,
                                  mode=config.mode_reduce_plteau
                                  )
    
    return [checkpoint, earlystop, reduce_lr]



def define_snn_optimizer(config):
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.snn_lr,
                                         beta_1=config.snn_beta_1, 
                                         beta_2=config.snn_beta_2,
                                         epsilon=config.snn_epsilon,
                                         amsgrad=config.snn_amsgrad,
                                         )
    return optimizer



def define_mnn_optimizer(mnn_lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=mnn_lr)
    return optimizer


