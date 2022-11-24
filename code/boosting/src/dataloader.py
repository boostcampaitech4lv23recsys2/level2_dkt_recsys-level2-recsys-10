import pandas as pd 
import random

class MyDataLoader:

    def __init__( self,
                  train_file:str,
                  test_file:str)->None :

        self.data_path = {} 
        self.data_path["train_file"]  = train_file
        self.data_path["test_file"]   = test_file

        try:
            self.data = {}
            self.data["train"]        =  pd.read_csv(self.data_path["train_file"])
            self.data["test"]         =  pd.read_csv(self.data_path["test_file"])
        except Exception as e:
            print("Error ",e)


    def get_data(self) -> dict :
        
        return self.data

    # models.py 로 이동
    # def preprocess_data(self) -> dict :
        
    #     self.data["train"] = self.preprocessing_ft( self.data["train"] )
    #     self.data["test"]  = self.preprocessing_ft( self.data["test"] )
        
    #     return self.data
