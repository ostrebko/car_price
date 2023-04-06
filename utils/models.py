import pandas as pd

from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow.keras import Input
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model, Sequential



def define_cat_boost(config):

    """
    Function for define CatBoostRegressor model (an object of the CatBoostRegressor class)
   
    return: model - CatBoostRegressor model for foolwing train
    -------
    params:
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)

    """

    model = CatBoostRegressor(iterations=config.cat_iterations,
                              depth=config.cat_depth, 
                              learning_rate=config.cat_learning_rate,
                              random_seed=config.RANDOM_SEED,
                              eval_metric=config.cat_eval_metric,
                              custom_metric=[config.cat_custom_metric_1, 
                                             config.cat_custom_metric_2],
                              od_wait=config.cat_od_wait,
                              grow_policy=config.cat_grow_policy,
                              l2_leaf_reg=config.cat_l2_leaf_reg,
                              model_size_reg=config.cat_model_size_reg,
                              loss_function=config.cat_loss_function #'MAE'
                              )
    return model


class MyNaiveModel():
    
    """
    Function to get predict with "naive" model 
        to compare other models with it.

    return: calculated (predicted) list of target 'price' df_test
    -------
    params:
    
    data_train - DataFrame of train samples
    data_test - DataFrame of test samples
    
    """ 
    
    def __init__(self, data_train, data_test):
        
        self.data_train = data_train
        self.data_test = data_test
        self.predicts = self.get_predict_naive_model()


    def get_predict_naive_model(self):
        
        predicts = []
        for index, row in pd.DataFrame(self.data_test[['model_info', 'productionDate']]).iterrows():
            query = f"model_info == '{row[0]}' and productionDate == {row[1]}"
            predicts.append(self.data_train.query(query)['price'].median())

        # filling in the matches that were not found by the median values
        predicts = pd.DataFrame(predicts, columns=['price'])
        predicts = predicts.fillna(predicts.median())

        # rounding
        predicts = (predicts // 1000) * 1000
        
        return predicts
    

class SimpleSeqNN(Model):

    def __init__(self, config: dict):
        super().__init__()
        self.is_show_summary = config.is_show_summary
        self.config = config
        
        self.inputs = x = Input(shape=(self.config.snn_input_shape,))
        x = L.Dense(self.config.snn_dense_units_1, 
                    activation=self.config.snn_dense_activation,
                    name='dense_layer_1')(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(0.5)(x)
        x = L.Dense(self.config.snn_dense_units_2, 
                    activation=self.config.snn_dense_activation,
                    name='dense_layer_2')(x)
        x = L.Dropout(0.25)(x)
        x = L.Dense(self.config.snn_dense_units_3, 
                    activation=self.config.snn_dense_activation,
                    name='dense_layer_3')(x)
        x = L.Dropout(0.5)(x)
        self.outputs = L.Dense(1, 
                               activation=self.config.snn_output_activation)(x)
        
        
    def build_model(self):
        """
        Model formation method
        """
        model = Model(
            inputs=self.inputs,
            outputs=self.outputs,
            name="model_SSNN"
        )
        if self.is_show_summary:
            model.summary()
        return model
    


class MultiSeqNN(Model):

    def __init__(self, config: dict, tokenize):
        super().__init__()
        self.is_show_summary = config.is_show_summary
        self.config = config
        
        self.inputs_1 = x1 = Input(shape=(self.config.MAX_SEQUENCE_LENGTH,), 
                                          name=config.mnn_nlp_name_1)
        x1 = L.Embedding(len(tokenize.word_index)+1, 
                            config.MAX_SEQUENCE_LENGTH,)(x1)
        x1 = L.BatchNormalization()(x1)
        x1 = L.LSTM(config.mnn_lstm_units_1, 
                       return_sequences=config.mnn_return_seq_1)(x1)
        x1 = L.Dropout(0.25)(x1)
        x1 = L.LSTM(config.mnn_lstm_units_2)(x1)
        x1 = L.Dropout(0.5)(x1)
        x1 = L.Dense(self.config.mnn_nlp_dense_units_1, 
                     activation=self.config.mnn_nlp_dense_activation,
                     )(x1)
        model_nlp = L.Dropout(0.5)(x1)


        self.inputs_2 = x2 = Input(shape=(self.config.mnn_input_shape,))
        x2 = L.Dense(self.config.mnn_mlp_dense_units_1, 
                     activation=self.config.mnn_mlp_dense_activation_1,
                     )(x2)
        x2 = L.BatchNormalization()(x2)
        x2 = L.Dropout(0.25)(x2)
        x2 = L.Dense(self.config.mnn_mlp_dense_units_2, 
                    activation=self.config.mnn_mlp_dense_activation_2,
                    )(x2)
        model_mlp = L.Dropout(0.5)(x2)

        combinedInput = L.concatenate([model_nlp, model_mlp]) #.output, .output
        
        # being our regression head
        self.head = L.Dense(config.mnn_head_dense_units_1, 
                            activation=config.mnn_head_dense_activation_1
                            )(combinedInput)
        #self.head = L.BatchNormalization()(self.head)
        self.head = L.Dropout(0.5)(self.head)
        self.head = L.Dense(1, 
                            activation=config.mnn_output_activation
                            )(self.head)
        
        
    def build_model(self):
        """
        Model formation method
        """
        model = Model(
            inputs=[self.inputs_1, self.inputs_2],
            outputs=self.head,
            name="model_MultiSeqNN"
        )
        if self.is_show_summary:
            model.summary()
        return model
    
