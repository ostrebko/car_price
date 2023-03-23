import os
import re
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from IPython.display import display

# increasing of default size of graphs
from pylab import rcParams
rcParams['figure.figsize'] = 8, 4



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



def ch_engineDisplacement(d_frame_engine_d):
    """
    Function to bring 1 item of the engineDisplacement col to float value
    Uses with fuction: some_df.d_frame_engine_d.apply(ch_name_col)
    return: None
    -------
    params:
    df_name_item - item from 1 row of col 'engineDisplacement'
    
    """
    return float(d_frame_engine_d[:-4])



def numb_type_analisys(df, col_name, bins_step=1):

    """
    Function to print statistical analisys of numerical features
    Describe statistical information about data of col_name
        and plot histogram

    return: float value
    -------
    params:
    df - DataFrame (df_train or df_test)
    col_name - name of numerical feature from df
    bins_step - bins step in histogram

    """

    print('\n', '\033[1m' + 'Column: ' + col_name, '\033[0m', '\n')

    IQR = df[col_name].quantile(0.75) - df[col_name].quantile(0.25)
    perc25 = df[col_name].quantile(0.25)
    perc75 = df[col_name].quantile(0.75)
    out_left = perc25 - 1.5*IQR
    out_right = perc75 + 1.5*IQR

    print('Statistical parameters of the column {}:'.format(col_name))
    print(
        '25-й percentile: {},'.format(perc25),
        '75-й percentile: {},'.format(perc75),
        "IQR: {}, ".format(IQR),
        "Boundaries outliers: [{f}, {l}].".format(f=out_left, l=out_right),
        '\n')

    if out_left > min(df[col_name]):
        print('There are outliers in the region of minimum values')
    if out_right < max(df[col_name]):
        print('There are outliers in the region of maximum values')
    if (out_left < min(df[col_name])) and (out_right > max(df[col_name])):
        print('There are no outliers')

    display(df[col_name].describe())

    df[col_name].hist(bins=np.arange(min(df[col_name]), max(df[col_name]) + 1, bins_step),
                      align='left',
                      label=col_name)
    plt.legend()