import wandb 
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from .dataloader import * 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import os 
import numpy as np

# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
def custom_train_test_split(df, FEATS, orient_key = 'userID',ratio=0.7, split=True):
    
    users = list(zip(df[orient_key].value_counts().index, df[orient_key].value_counts()))
    random.shuffle(users)
    
    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df[orient_key].isin(user_ids)]
    valid = df[df[orient_key].isin(user_ids) == False]

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    valid = valid[valid['userID'] != valid['userID'].shift(-1)]

    train_X = train.drop("answerCode",axis = 1 )
    y_train = train["answerCode"]

    valid_X = valid.drop("answerCode",axis = 1 )
    y_valid = valid["answerCode"]

    return train_X[FEATS], y_train, valid_X[FEATS], y_valid


class MLModelBase:

    def __init__(self, 
                 data_collection :dict, 
                 best_params : dict, 
                 preprocessing_ft,
                 use_feat_list : list ):
        
        self.best_params = best_params 
        self.data = data_collection
        self.model = XGBClassifier( **self.best_params)
        # 사용할 Feature 설정
        self.FEATS = use_feat_list
        
        self.data["train"] = preprocessing_ft( self.data["train"] )
        self.data["test"]  = preprocessing_ft( self.data["test"] )
        
        # split 변경이 필요하면 각 클래스에 포함 시켜줘야 함 
        self.train_X, self.y_train, self.valid_X, self.y_valid \
            = custom_train_test_split( self.data["train"],self.FEATS )

    
    # def preprocessing(self):
    #     self.data["train"] = self.preprocessing_ft( self.data["train"] )
    #     self.data["test"]  = self.preprocessing_ft( self.data["test"] )
        
    #     return self.data

    def train(self):

        wandb.login()
        wandb.init(project = "DEFAULT:XGBC", config = self.best_params)

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

        return auc

    def inference(self):

        # LEAVE LAST INTERACTION ONLY
        test_df = self.data["test"][self.data["test"]['userID'] != self.data["test"]['userID'].shift(-1)]

        # DROP ANSWERCODE
        y_test = test_df['answerCode']
        test_X = test_df.drop(['answerCode'], axis=1)

        self.total_preds = self.model.predict(test_X[self.FEATS])

        return self.total_preds


class MyXGBoostClassifier(MLModelBase) :

    def __init__(self, data_collection :dict, 
                       best_params : dict, 
                       preprocessing_ft, 
                       use_feat_list : list ):
        
        # default data split 방법은 custom data split 이다. 
        super().__init__( data_collection, best_params, preprocessing_ft,use_feat_list)

        self.model = XGBClassifier( **self.best_params)

    def train(self):

        wandb.login()
        wandb.init(project = "XGBC", config = self.best_params)

        print(self.train_X)
        self.model.fit( self.train_X[self.FEATS], self.y_train,
                        eval_set=[(self.valid_X[self.FEATS], self.y_valid)],
                        eval_metric='auc',
                        verbose=50,
                        early_stopping_rounds= 50)

        y_pred_train = self.model.predict_proba( self.train_X[self.FEATS] )[:,1]
        y_pred_valid = self.model.predict_proba(self.valid_X[self.FEATS])[:,1]

        # make predictions on test
        acc = accuracy_score(self.y_valid, np.where(y_pred_valid >= 0.5, 1, 0))
        auc = roc_auc_score( self.y_valid,y_pred_valid)

        wandb.log({"valid_accuracy": acc})
        wandb.log({"valid_roc_auc": auc})

        return auc


class MyLGBM(MLModelBase):

    def __init__(self, data_collection :dict, 
                    best_params : dict, 
                    preprocessing_ft, 
                    use_feat_list : list ):
    
        # default data split 방법은 custom data split 이다. 
        super().__init__( data_collection, best_params, preprocessing_ft,use_feat_list)
        
        self.lgb_train = lgb.Dataset( self.train_X[self.FEATS], self.y_train)
        self.lgb_valid = lgb.Dataset( self.valid_X[self.FEATS], self.y_valid)
        self.model = lgb

    def train(self):

        wandb.init(project="LGBM", config= self.best_params)
        
        self.model = self.model.train(
            self.best_params, 
            self.lgb_train,
            valid_sets=[self.lgb_train, self.lgb_valid],
            verbose_eval=100,
            num_boost_round=500,
            early_stopping_rounds=100,
            valid_names=('validation'),
            callbacks=[wandb.lightgbm.wandb_callback()] )

        wandb.lightgbm.log_summary(self.model, save_model_checkpoint=True)

        preds = self.model.predict(self.valid_X[self.FEATS])

        acc = accuracy_score(self.y_valid, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(self.y_valid, preds)
        wandb.log({"valid_accuracy": acc})
        wandb.log({"valid_roc_auc": auc})

        return auc

class MyLGBMClassifier(MLModelBase):

    def __init__(self, data_collection :dict, 
                    best_params : dict, 
                    preprocessing_ft, 
                    use_feat_list : list ):
    
        # default data split 방법은 custom data split 이다. 
        super().__init__( data_collection, best_params, preprocessing_ft,use_feat_list)
        
        self.lgb_train = lgb.Dataset( self.train_X[self.FEATS], self.y_train)
        self.lgb_valid = lgb.Dataset( self.valid_X[self.FEATS], self.y_valid)
        self.model = LGBMClassifier( **self.best_params)

    def train(self):

        wandb.init(project="LGBMClassifier", config= self.best_params)
        
        self.model.fit(
                    X=self.train_X[self.FEATS],
                    y=self.y_train,
                    eval_set=[(self.valid_X[self.FEATS], self.y_valid)],
                    early_stopping_rounds=100,
                    verbose=20,
                )

        # wandb.lightgbm.log_summary(self.model, save_model_checkpoint=True)
        preds = self.model.predict(self.valid_X[self.FEATS])

        acc = accuracy_score(self.y_valid, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(self.y_valid, preds)
        wandb.log({"valid_accuracy": acc})
        wandb.log({"valid_roc_auc": auc})

        return auc


class MyCatClassifier(MLModelBase):

    def __init__(self, data_collection :dict, 
                    best_params : dict, 
                    preprocessing_ft, 
                    use_feat_list : list ):
    
        # default data split 방법은 custom data split 이다. 
        super().__init__( data_collection, best_params, preprocessing_ft,use_feat_list)
        
        self.model = CatBoostClassifier( **self.best_params )
                    # loss_function='CrossEntropy' 
        self.cat_features = [f for f in self.train_X.columns if self.train_X[f].dtype == 'object' or self.train_X[f].dtype == 'category']
        
    def train(self):

        wandb.login()    
        wandb.init(project = "CatBC", config = self.best_params)

        self.model.fit(
            self.train_X, self.y_train,
            cat_features=self.cat_features,
            eval_set=(self.valid_X, self.y_valid),
            verbose=False
        )

        preds = self.model.predict(self.valid_X[self.FEATS])

        acc = accuracy_score(self.y_valid, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(self.y_valid, preds)
        wandb.log({"valid_accuracy": acc})
        wandb.log({"valid_roc_auc": auc})

        return auc

