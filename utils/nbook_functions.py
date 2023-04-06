import numpy as np
import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import PIL


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
        # If NaN - convert to 0
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
    Function for preprocessing tabular data (DataFrame)
    for following splitting to train model
   
    return: df_output - preprocessed tabular data
    -------
    params:
    df_input - concatenated df_train and df_test DataFrames
    
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
    Function to concatenate df_train and df_test for a single preprocessing.
    df_train and df_test are read from csv files
   
    return: df_train and df_test concatenated DataFrame
    -------
    params:
    paths - dict of paths which created from create_paths func

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
    Function to proove preprocessing data
   
    return: X_sub, X_train, X_test, y_train, y_test - splitted data for train model
    -------
    params:
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    paths - dict of paths which created from create_paths func

    """

    data = concat_train_test(paths)
    df_preproc = preproc_features(data)
    print('tabular preproc done')
    
    train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
    test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

    y = train_data.price.values     # таргет
    X = train_data.drop(['price'], axis=1)
    X_sub = test_data.drop(['price'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.13, shuffle=True, random_state=config.RANDOM_SEED)
    
    print('data preproc finish')
    
    return X_sub, X_train, X_test, y_train, y_test # X, y, 


def plot_history_loss(history):
    
    """
    Function to ...
   
    return: None
    -------
    params:
    history - ...

    """


    plt.title('Loss')
    plt.plot(history.history['MAPE'], label='train')
    plt.plot(history.history['val_MAPE'], label='test')
    plt.legend(labels=['train','test'], loc='upper right')
    plt.show();



def save_best_model(model, paths, 
                    best_trained_weights,
                    best_weights_to_save,
                    best_model_to_save):
    
    """
    Function to # load best weights model and save model
   
    return: None
    -------
    params:
    model - ...
    paths - ...
    best_trained_weights - ...
    best_weights_to_save - ...
    best_model_to_save - ...
    """

    
    model.load_weights(os.path.join(paths.PATH_BEST_MODEL, best_trained_weights))
    model.save_weights(os.path.join(paths.PATH_MODELS, best_weights_to_save))
    model.save(os.path.join(paths.PATH_MODELS, best_model_to_save))
    print('best weights and model of Simple Sequential Model saved')



def show_sample_imgs(paths, n_samples):
    
    """
    Function to # load best weights model and save model
   
    return: None
    -------
    params:
    
    paths - ...
    
    """
    
    train = pd.read_csv(os.path.join(paths.PATH_DATA, 'train.csv'))

    random_image = train.sample(n=n_samples)
    random_image_paths = random_image['sell_id'].values
    random_image_cat = random_image['price'].values

    if n_samples > 1:
        
        plt.figure(figsize = (12,9))
        
        for index, path in enumerate(random_image_paths):
            im = PIL.Image.open(os.path.join(paths.PATH_DATA_IMGS, str(path)+'.jpg'))
            plt.subplot(3, 3, index+1)
            plt.imshow(im)
            plt.title('price: ' + str(random_image_cat[index]))
            plt.axis('off')
        plt.show()

    else:
        plt.figure(figsize = (4,3))
        im = PIL.Image.open(os.path.join(paths.PATH_DATA_IMGS, 
                                         str(random_image_paths[0])+'.jpg'))
        plt.imshow(im)
        plt.title('price: ' + str(random_image_cat[0]))
        plt.axis('off')
        plt.show()
        print(im.size)
        

    