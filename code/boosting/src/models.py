import wandb 
from xgboost import XGBClassifier
from .dataloader import * 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import os 
import numpy as np

class MyXGBoostClassifier :

    def __init__(self, data_collection :dict, best_params : dict ):
        
        """
        booster=config.booster, 
        max_depth=config.max_depth,
        learning_rate=config.learning_rate, 
        subsample=config.subsample 
        """

        self.best_params = best_params 
        self.data = data_collection
        self.model = XGBClassifier( **self.best_params)

        self.train_X, self.y_train, self.valid_X, self.y_valid = custom_train_test_split( self.data["train"] )
    
    def train(self):

        wandb.login()
        wandb.init(project = "yujin", config = self.best_params)

        print(self.train_X)
        self.model.fit( self.train_X, self.y_train,
                        eval_set=[(self.valid_X, self.y_valid)],
                        eval_metric='auc',
                        verbose=50,
                        early_stopping_rounds= 50)

        y_pred_train = self.model.predict_proba( self.train_X )[:,1]
        y_pred_valid = self.model.predict_proba(self.valid_X)[:,1]

        # make predictions on test
        acc = accuracy_score(self.y_valid, np.where(y_pred_valid >= 0.5, 1, 0))
        auc = roc_auc_score( self.y_valid,y_pred_valid)

        wandb.log({"valid_accuracy": acc})
        wandb.log({"valid_roc_auc": auc})


    def inference(self):
        # LOAD TESTDATA
        # test_csv_file_path = os.path.join(data_dir, 'test_data.csv')
        # test_df = pd.read_csv(test_csv_file_path)

        # # FEATURE ENGINEERING
        # test_df = feature_engineering(test_df)

        # LEAVE LAST INTERACTION ONLY
        test_df = self.data["test"][self.data["test"]['userID'] != self.data["test"]['userID'].shift(-1)]

        # DROP ANSWERCODE
        y_test = test_df['answerCode']
        test_X = test_df.drop(['answerCode'], axis=1)

        self.total_preds = self.model.predict(test_X)

    def save():

        # SAVE OUTPUT
        output_dir = 'output/'
        write_path = os.path.join(output_dir, "submission.csv")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(self.total_preds):
                w.write('{},{}\n'.format(id,p))


