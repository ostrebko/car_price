import os
from dotmap import DotMap



def create_paths(config, is_in_root=False):
    
    """
    Create paths to read, save and load data, models for train and inference
    
    -------
    params:
    
    config - dict (Dotmap) from configuration file with defined parameters values 
             (creates from config_reader function by reading data_config.json)
    is_in_root=True for main.py inference or run app .py in root dir or 
    is_in_root=False for notebook train
    
    """
    
    paths_dict = dict()

    if is_in_root:
        paths_dict['PATH_DATA'] = config.PATH_DATA
        paths_dict['PATH_MODELS'] = config.folder_models
    else: 
        paths_dict['PATH_DATA'] = os.path.join('..', config.PATH_DATA)
        paths_dict['PATH_MODELS'] = os.path.join('..', config.folder_models)

    paths_dict['PATH_DATA_IMGS'] = os.path.join(paths_dict['PATH_DATA'], 
                                                 config.folder_imgs)
    paths_dict['PATH_OUTPUTS'] = os.path.join(paths_dict['PATH_DATA'], 
                                              config.folder_outputs)
    paths_dict['PATH_BEST_MODEL'] = os.path.join(paths_dict['PATH_DATA'], 
                                                 config.folder_best_model) 
    paths_dict['PATH_TO_ZIP'] = os.path.join(paths_dict['PATH_DATA'], 
                                             config.ZIP_FILE)
    paths_dict['PATH_TO_TEST_PREDICTIONS'] = os.path.join(paths_dict['PATH_DATA'], 
                                                          config.test_imgs_for_pred)
    paths = DotMap(paths_dict)
    
    return paths
