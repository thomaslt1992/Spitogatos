import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_dataset_for_training(df, target_col,split=False,p=0.2,na=True):

    if na:
        df = df.dropna(axis=0)

    df = df[df['price'] > 10].reset_index()


    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # One-hot encode string columns
    object_cols = list(X.select_dtypes(include='object').columns)
    if object_cols:
        X = pd.get_dummies(X, columns=object_cols)
    
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p, random_state=21)
        return X_train,X_test,y_train,y_test
    else:
        return X,y