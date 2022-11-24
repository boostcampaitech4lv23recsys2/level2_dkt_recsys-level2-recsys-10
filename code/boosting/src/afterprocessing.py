import pandas as pd
from sklearn.preprocessing import LabelEncoder


def mapping_cat_to_label(df:pd.DataFrame):

    for col in df.columns : 
        if df[col].dtype not in ['int', 'float', 'bool']:
            encoder = LabelEncoder()
            encoder.fit(df[col])
            df[col] = encoder.transform(df[col])

    return df
