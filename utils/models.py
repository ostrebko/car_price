import pandas as pd

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