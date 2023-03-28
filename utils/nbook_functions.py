import numpy as np
import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer



def mape_func(y_true, y_pred):
    """
    Function to calc MAPE-metric

    return: calculated metric MAPE
    -------
    params:
    
    y_true - there are known values of target y
    y_pred - predicted values of target y
    """
    return np.mean(np.abs((y_pred - y_true)/y_true))



def prepare_vladenie(string_name, col='Владение', pattern_1 = '\d+'):
    
    """
    Function to prepare feature 'Владение' to num type feature
    if NaN -> value changes to 0
    elif 'и' in string value (ex:'1 год и 1 мес)' -> 
        -> value recalc int(month_num)+12*int(year_num)
    elif 'мес' in string value (ex:'8 месяцев)' -> 
        -> value recalc int(month_num)
    esle ex:'1 год' -> value recalc 12*int(year_num)
    
    return: number of how many months of car ownership 
    -------
    params:
    string_name - 1 value of feature 'Владение' in DataFrame
    
    """
        
    if type(string_name) == float:
        # Если NaN меняем на 0
        num_mounth = 0
    elif len(string_name.split('и'))==2:
        nums = re.findall(pattern_1, string_name)
        num_mounth = int(nums[1]) + int(nums[0])*12
    elif 'мес'in string_name:
        num_mounth = int(re.findall(pattern_1, string_name)[0])
    else: 
        num_mounth = int(re.findall(pattern_1, string_name)[0])*12
    return num_mounth



def ch_name_col(df_name_item):
    
    """
    Function to exclude duplicated data from 1 row of col 'name'
    Uses with fuction: some_df.df_name_item.apply(ch_name_col)
    return: changed column name
    -------
    params:
    df_name_item - item from 1 row of col 'name'
    
    """

    if '4WD' in df_name_item:
        full_drive = ' 4WD'
    else:
        full_drive = ''
    pattern_1 = ' \d\.\d'
    pattern_2 = '\d\.\d'
    pattern_3 = ' AT'
    if len(re.findall(pattern_1, df_name_item))!=0:
        return re.split(pattern_1, df_name_item)[0] + full_drive
    elif len(re.findall(pattern_2, df_name_item))!=0:
        return 'no_val' + full_drive
    else:
        return re.split(pattern_3, df_name_item)[0] + full_drive



def preproc_features(df_input):
    
    """
    Function to ...
   
    return: ....
    -------
    params:
    df_input - concatenated dataframe with train and test
    
    """
    
    df_output = df_input.copy()
    
    # 1. Preprocessing ----------------------------------------
    # unnecessary features for the model
    df_output.drop(['description','vehicleConfiguration', 'Руль', 'sell_id'], axis = 1, inplace=True)
    
    
    # Converting categorical features to numeric features
    df_output['ch_vladenie'] = df_output.Владение.apply(prepare_vladenie)
    df_output['enginePower'] = df_output.enginePower.apply(lambda x: int(x[:-4]))
    
    df_output.engineDisplacement.replace('undefined LTR', '0.0 LTR', inplace=True)
    df_output['engineDisplacement'] = df_output.engineDisplacement.apply(lambda x: float(x[:-4]))
    
    
    # Converting numerical features (sqrt)
    df_output['sqrt_modelDate'] = df_output['modelDate'].apply(np.sqrt)
    
    
    # Converting numerical features (log)
    df_output['log_mileage'] = df_output['mileage'].apply(lambda x: np.log(1+x))
    df_output['log_modelDate'] = df_output['modelDate'].apply(lambda x: np.log(2020-x)) 
    df_output['log_productionDate'] = df_output['productionDate'].apply(lambda x: np.log(1+x))
    
    
    # Categorical features - shorting feature 'name'
    df_output['name'] = df_output.name.apply(ch_name_col)
    
    # 2. Numerical Features ----------------------------------------
    numerical_features = list(set(df_output.dtypes[df_output.dtypes!=object].index) - \
                              set(['price', 'sample'])) 
    
    
    # NAN processing (removing 1 line with NaN in train part - see EDA)
    df_output.dropna(subset=['Владельцы'], inplace=True)
    
    
    # Data normalization + Data size reduction
    scaler = StandardScaler()
    for column in numerical_features:
        df_output[column] = scaler.fit_transform(df_output[[column]])[:,0].astype('float32')
              
    
    # 3. Categorical Features ----------------------------------------
    categorical_features = list(
        set(df_output.dtypes[df_output.dtypes==object].index) - set(['Владение']))
    
    
    # Label Encoding
    for column in categorical_features:
        df_output[column] = df_output[column].astype('category').cat.codes
        df_output[column] = StandardScaler().fit_transform(
            np.array(df_output[column]).reshape(-1, 1))
     
    
    # One-Hot Encoding: в pandas есть готовая функция - get_dummies.
    df_output = pd.get_dummies(df_output, columns=categorical_features, dummy_na=False)
    
    
    # 4. Feature Engineering (There are new features did not improve the quality in catboost, 
    #                         so they were not included)
    
    # новый признак: есть ли NaN во 'Владении'? 
    df_output['is_nan_vladenie'] = np.where(df_output['Владение'].isna(), 0, 1)
        
    
    # Clean
    # убираем признаки которые еще не успели обработать, 
    df_output.drop(['Владение', 'modelDate'], axis = 1, inplace=True)
    
    return df_output



def concat_train_test(paths):
    """
    Function to ...
   
    return: ....
    -------
    params:
    ..... - ....
    
    """
    
    train = pd.read_csv(os.path.join(paths.PATH_DATA, 'train.csv'))
    test = pd.read_csv(os.path.join(paths.PATH_DATA, 'test.csv'))
    #sample_submission = pd.read_csv(os.path.join(config.PATH_DATA, 'sample_submission.csv'))
    
    # changing the outlier value from 999999 to 99999 (see 01_Data_EDA.ipynb)
    train.loc[train['mileage']==train['mileage'].max(), 'mileage'] = 99999

    # changing the outlier value from 999999 to 99999 (see 01_Data_EDA.ipynb)
    test.loc[test['mileage']==test['mileage'].max(), 'mileage'] = 100000
        
    # For the correct processing of features, we combine the train and the test into one dataset
    train['sample'] = 1 # marking the train
    test['sample'] = 0 # marking the test
    test['price'] = 0 # there is no 'price' values in the test, so we fill it with zeros

    data = pd.concat([test, train]).reset_index(drop=True) 

    return data



def create_preproc_data(config, paths):
    
    """
    Function to ...
   
    return: ....
    -------
    params:
    ..... - ....
    
    """

    data = concat_train_test(paths)
    df_preproc = preproc_features(data)
    print('tabular preproc done')
    
    
    ## Cleaning data.decsription
    #data.description = get_clean_description(data.description, config)
    
    # Преобразование табличных данных функцией df_preproc
    train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
    test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

    y = train_data.price.values     # таргет
    X = train_data.drop(['price'], axis=1)
    X_sub = test_data.drop(['price'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.13, shuffle=True, random_state=config.RANDOM_SEED)
    
    list_inds = [X_train.index, X_test.index, X_sub.index]
    
    print('data preproc finish')
    
    return X_sub, X_train, X_test, y_train, y_test, list_inds # X, y, 


def define_cat_boost(config):
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
                                                






def clean_stopwords(str_2_clean):

    """
    Function to Функция для очистки данных от цифр, знаков, повторяющихся символов
   
    return: ....
    -------
    params:
    ..... - ....
    
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
    Function to ### делаем список для фильтра часто и редко встречающихся слов
   
    return: ....
    -------
    params:
    ..... - ....
    
    """
    
    filtered_dict = dict(
        filter(lambda item: (item[1]<max_freq_to_remove)or(item[1]>min_freq_to_remove), 
               dict_from_data.items()))
    return filtered_dict


def get_clean_description(data_descr, config):

    """
    Function to Функция для очистки данных от цифр, знаков, повторяющихся символов
   
    return: ....
    -------
    params:
    ..... - ....
    
    """

    # Cleaning from stopwords
    data_descr = data_descr.apply(clean_stopwords)
    print('clean stopwords of data.description done')
    
    tokenize = Tokenizer(num_words=config.MAX_WORDS)
    tokenize.fit_on_texts(data_descr)
    
    ## слова которые будем фильтровать
    filtered_values = set(filtetred_freq_words(100, 5000, tokenize.word_index).keys()) #(100, 9000)
    
    ## исключаем фильтрованные слова
    data_descr = data_descr.apply(
        lambda x: " ".join([word for word in x.split() if word not in filtered_values]))

    return data_descr



# Функция преобрабования числовых данных в вектора
def data_descr_to_nlp(data_descr, config, list_inds):    
    
    ## Токенизируем очищенное и преобразованное описание data description
    tokenize = Tokenizer(num_words=config.MAX_WORDS)
    tokenize.fit_on_texts(data_descr)
    
    # split данных
    text_train = data_descr.iloc[list_inds[0]]
    text_test = data_descr.iloc[list_inds[1]]
    text_sub = data_descr.iloc[list_inds[2]]
    
    text_train_sequences = sequence.pad_sequences(
        tokenize.texts_to_sequences(text_train), maxlen=config.MAX_SEQUENCE_LENGTH)
    text_test_sequences = sequence.pad_sequences(
        tokenize.texts_to_sequences(text_test), maxlen=config.MAX_SEQUENCE_LENGTH)
    text_sub_sequences = sequence.pad_sequences(
        tokenize.texts_to_sequences(text_sub), maxlen=config.MAX_SEQUENCE_LENGTH)
    
    return text_train_sequences, text_test_sequences, text_sub_sequences, tokenize