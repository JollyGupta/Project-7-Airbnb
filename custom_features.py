import pandas as pd
from datetime import datetime

#def add_custom_features(X):
 #   X = X.copy()
#    X['host_experience_years'] = datetime.today().year - pd.to_datetime(X['host_since'], errors='coerce').dt.year
#    X['host_experience_years'] = X['host_experience_years'].fillna(0).astype(int)
#    return X.drop(columns=['host_since'])

# If host_since is not present in the input DataFrame, this function will crash with a KeyError.


def add_custom_features(X):
    if 'host_since' in X.columns:
        X = X.copy()
        X['host_experience_years'] = (
            datetime.today().year - pd.to_datetime(X['host_since'], errors='coerce').dt.year
        )
    return X
