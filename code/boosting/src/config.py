# ====================================================
# CFG 
# ====================================================

boosting_params = {
    "CatBoostClassifier" : {
            "learning_rate"     : 0.048, 
            "iterations"        : 5,
            "random_state"      : 42 
            
    },
  
    "LGBM" : {
        'objective': "binary",# regression, binary, multiclass
        'learning_rate':1e-2,
        'drop_rate':0.1,
        'num_iterations':500
    },

    "LGBMClassifier" : {
        "max_depth" : 6,
        "num_leaves" : 60,
        # min_data_in_leaf = 10,
        "seed" : 42,
        "metric" : "auc",
        "objective" : "binary",
        "n_estimators" : 200
    },

    "XGBClassifier":{
        "booster": "gbtree",
        "subsample": 1,
        "seed": 42,
        "colsample_bytree" : 0.3,
        "learning_rate" : 0.01,
        "max_depth" : 15,
        "alpha" : 10,
        "n_estimators":5,
    }
}