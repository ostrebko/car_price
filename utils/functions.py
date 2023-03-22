import os
import re
import numpy as np
import zipfile


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



def make_unzip(path_to_zip_file, path_to_unzip, is_unzip=False):
       
    """
    Function for extracting data from zip-archive sf-dst-car-price-prediction-part2.zip
    and creates folders
    
    -------
    params:

    path_to_zip_file - path to zipfile
    path_to_unzip - path to unzip data
    
    """

    if is_unzip:
        # Extract zip-archive with all data
        print('Unzip sf-dst-car-price-prediction-part2.zip')
        
        # Extract to folder 'inputs_for_train' without subfolder 'sf-dl-car-classification'
        with zipfile.ZipFile(path_to_zip_file) as z_file:
            for zip_info in z_file.infolist():
                zip_info.filename = os.path.join(os.path.dirname(zip_info.filename).split('/')[-1],
                                                 os.path.basename(zip_info.filename))
                z_file.extract(zip_info, path_to_unzip)
        print('Successfully unziped "sf-dst-car-price-prediction-part2.zip"')

    else:
        print('Zip-archive "sf-dst-car-price-prediction-part2.zip" no need to unpack,',
              'to unpack change value of param "is_unzip" to "True"')

 
def get_translite(text_string):
    
    """
    Function to get translite from russian word to english word

    return: transliterated to english text_string
    
    -------
    params:

    text_string: word (string) without spaces
    
    """
    
    symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ ",
               (*list(u'abvgdee'), 'zh', *list(u'zijklmnoprstuf'), 'kh', 'z', 'ch', 'sh', 'sh', '',
               'y', '', 'e', 'yu','ya', *list(u'ABVGDEE'), 'ZH', 
               *list(u'ZIJKLMNOPRSTUF'), 'KH', 'Z', 'CH', 'SH', 'SH', *list(u'_Y_E'), 'YU', 'YA', ' '))

    coding_dict = {source: dest for source, dest in zip(*symbols)}
    my_simb_table = text_string.maketrans(coding_dict)
    
    return text_string.translate(my_simb_table).lower() 


def pre_an_types_feat(df_name):
    """
    Function to get preanalisys of features types
    i.e. binary, categorical, large_categorical, other
    
    return: names of features for each calculated type:
            feat_bin, feat_cat, feat_cat_large, feat_other
    -------
    params:

    df_name - pandas DataFrame
    
    """
    
    mask_binary = np.logical_and(df_name.nunique() > 0, df_name.nunique() <= 2)
    mask_cat = np.logical_and(df_name.nunique() > 2, df_name.nunique() <= 20)
    mask_cat_large = np.logical_and(df_name.nunique() > 20, df_name.nunique() <= 50)
    mask_num = df_name.nunique() > 50
    feat_bin = list(df_name.nunique()[mask_binary].index)
    feat_cat = list(df_name.nunique()[mask_cat].index)
    feat_cat_large = list(df_name.nunique()[mask_cat_large].index)
    feat_other = list(df_name.nunique()[mask_num].index)
    return feat_bin, feat_cat, feat_cat_large, feat_other



def prints_types_feat(feat_bin, feat_cat, feat_cat_large, feat_other):
    """
    Function to print of features types
    i.e. binary, categorical, large_categorical, other
    
    return: None
    -------
    params:
    feat_bin, feat_cat, feat_cat_large, feat_other - returns from pre_an_types_feat
    
    """
    print(f'Binary features: \n{feat_bin} \n','--'*20)
    print(f'Categorical features with 2 < num_cat <= 20: \n{feat_cat} \n','--'*20)
    print('Categorical or num features with unique num of values:',
          f'20 < num_cat <= 50: \n{feat_cat_large}, \n','--'*20)
    print(f'Num or other types of features with unique num of values > 50: \n{feat_other}')



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


# анализ, где мы можем ошибиться на тестовых данных при обучении
def col_preanalisys(df_train, df_test, col, is_print_unique=False):
    
    """
    Function to compare unique test data with unique train data in one column
    Print conclusions and unique test data which is not in train data column
    
    return: None 
    -------
    params:
    df_train, df_test - train and test DataFrames
    col - column name for review
    is_print_unique - Print or not all unique test data which is not in train data column
    
    """
    unq_in_trn_col = set(df_train[col].unique())
    unq_in_tst_col = set(df_test[col].unique())
    unq_dlt_tst_col = unq_in_tst_col - unq_in_trn_col

    if (len(unq_in_trn_col)>=len(unq_in_tst_col))and(len(unq_dlt_tst_col)!=0):
        print(f'The number of unique values in the {col} column in df_train is greater or equal than in df_test.\n', 
              f'The number of {col} values in test that are not present in train: {len(unq_dlt_tst_col)}')
        if is_print_unique:
            print(f'{col} values in test that are not present in train: {unq_dlt_tst_col}')
        print('--'*20)
    elif (len(unq_in_trn_col)<len(unq_in_tst_col))and(len(unq_dlt_tst_col)!=0):
        print(f'The number of unique values in the {col} column in df_test is greater or equal than in df_train.\n',
              f'The number of {col} values in test that are not present in train: {len(unq_dlt_tst_col)}')
        if is_print_unique:
            print(f'{col} values in test that are not present in train: {unq_dlt_tst_col}')

        print('--'*20)
