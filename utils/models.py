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

        # ---------- TABULAR ----------
        
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
        
        # ---------- NLP part ----------

        self.nlp_inputs = x1 = Input(shape=(self.config.MAX_SEQUENCE_LENGTH,), 
                                            name=config.msnn_nlp_name_1)
        x1 = L.Embedding(len(tokenize.word_index)+1, 
                         config.MAX_SEQUENCE_LENGTH,)(x1)
        x1 = L.BatchNormalization()(x1)
        x1 = L.LSTM(config.msnn_lstm_units_1, 
                       return_sequences=config.msnn_return_seq_1)(x1)
        x1 = L.Dropout(0.5)(x1) #new 0.25->0.5
        x1 = L.LSTM(config.msnn_lstm_units_2)(x1)
        x1 = L.Dropout(0.5)(x1)
        x1 = L.Dense(self.config.msnn_nlp_dense_units_1, 
                     activation=self.config.msnn_nlp_dense_activation,
                     )(x1)
        model_nlp = L.Dropout(0.5)(x1)

        # ---------- TABULAR part ----------

        self.tab_inputs = x2 = Input(shape=(self.config.msnn_input_shape,))
        x2 = L.Dense(self.config.msnn_tab_dense_units_1, 
                     activation=self.config.msnn_tab_dense_activation_1,
                     )(x2)
        x2 = L.BatchNormalization()(x2)
        x2 = L.Dropout(0.5)(x2) #new  0.25->0.5
        x2 = L.Dense(self.config.msnn_tab_dense_units_2, 
                    activation=self.config.msnn_tab_dense_activation_2,
                    )(x2)
        model_tab = L.Dropout(0.5)(x2)

        # combining the outputs of two neural networks
        combinedInput = L.concatenate([model_nlp, model_tab]) 
        

        # ---------- HEAD of models ----------

        # being our regression head
        self.head = L.Dense(config.msnn_head_dense_units_1, 
                            activation=config.msnn_head_dense_activation_1
                            )(combinedInput)
        #self.head = L.BatchNormalization()(self.head) # back
        self.head = L.Dropout(0.5)(self.head)
        self.head = L.Dense(1, 
                            activation=config.msnn_output_activation
                            )(self.head)
        
        
    def build_model(self):
        """
        Model formation method
        """
        model = Model(
            inputs=[self.nlp_inputs, self.tab_inputs],
            outputs=self.head,
            name="model_MultiSeqNN"
        )
        if self.is_show_summary:
            model.summary()
        return model
    




class MultiInputNN(Model):

    def __init__(self, config: dict, tokenize):
        super().__init__()
        self.is_show_summary = config.is_show_summary
        self.config = config

        # ---------- IMAGES part ----------
        self.base_img_model = tf.keras.applications.efficientnet.EfficientNetB3(
            weights=config.minn_weights,
            include_top=config.minn_include_top,
            input_shape = (config.img_height, config.img_weight, config.img_channels)
            )
        
        self.base_img_model.trainable = config.minn_base_model_trainable

        # Сhoose layers which weights will train and freeze  
        if config.minn_train_all_base_layers==False:
            # Fine-tune from this layer onwards
            fine_tune_at = int(len(self.base_img_model.layers)//config.minn_f_tune_coef)
            # Freeze all the layers before the `fine_tune_at` layer
            for layer in self.base_img_model.layers[:fine_tune_at]:
                layer.trainable = False
        
        x0 = L.GlobalAveragePooling2D()(self.base_img_model.output)
        x0 = L.Dense(config.minn_imgs_dense_units_1, 
                     activation=config.minn_imgs_dense_activation_1)(x0)
        model_img = L.Dropout(0.5)(x0)
        

        # ---------- TABULAR part ----------
        self.tab_input = x1 = Input(shape=(self.config.minn_tab_input_shape,))
        x1 = L.Dense(self.config.minn_tab_dense_units_1, 
                     activation=self.config.minn_tab_dense_activation_1,
                     )(x1)
        #x1 = L.BatchNormalization()(x1)
        x1 = L.Dropout(0.25)(x1)
        x1 = L.Dense(self.config.minn_tab_dense_units_2, 
                    activation=self.config.minn_tab_dense_activation_2,
                    )(x1)
        model_tab = L.Dropout(0.5)(x1)

        
        # ---------- NLP part ----------
        self.nlp_input = x2 = Input(shape=(self.config.MAX_SEQUENCE_LENGTH,), 
                                           name=config.minn_nlp_name_1)
        x2 = L.Embedding(len(tokenize.word_index)+1, 
                             config.MAX_SEQUENCE_LENGTH,)(x2)
        #x2 = L.BatchNormalization()(x2)
        x2 = L.LSTM(config.minn_lstm_units_1, 
                       return_sequences=config.minn_return_seq_1)(x2)
        x2 = L.Dropout(0.25)(x2)
        x2 = L.LSTM(config.minn_lstm_units_2)(x2)
        x2 = L.Dropout(0.5)(x2)
        x2 = L.Dense(self.config.minn_nlp_dense_units_1, 
                     activation=self.config.minn_nlp_dense_activation,
                     )(x2)
        model_nlp = L.Dropout(0.5)(x2)


        # combining the outputs of three neural networks
        combinedInput = L.concatenate([model_img, model_tab, model_nlp])


        # ---------- HEAD of models ----------

        # being our regression head
        self.head = L.Dense(config.minn_head_dense_units_1, 
                            activation=config.minn_head_dense_activation_1
                            )(combinedInput)
        self.head = L.BatchNormalization()(self.head)
        self.head = L.Dropout(0.5)(self.head)
        self.head = L.Dense(1, 
                            activation=config.minn_output_activation
                            )(self.head)
        
        
    def build_model(self):
        """
        Model formation method
        """
        model = Model(
            inputs=[self.base_img_model.input, 
                    self.tab_input, 
                    self.nlp_input],
            outputs=self.head,
            name="model_MultiInputNN"
        )
        if self.is_show_summary:
            model.summary()
        return model