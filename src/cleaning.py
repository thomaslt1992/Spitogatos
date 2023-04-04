import pandas as pd
from pandas.api.types import is_numeric_dtype

THRESHOLD = 80
OUTLIER_THRESHOLD = 3
MINIMUM_ENTRIES_ACCEPTED = 20

def remove_nan(df,threshold = THRESHOLD): 

    na_df = pd.DataFrame(df.isna().sum()/len(df)*100,columns=['na_percentage'])
    return df.drop(columns=na_df[na_df['na_percentage']>threshold].index).copy()


def remove_outlier(df, column_name, threshold=OUTLIER_THRESHOLD):

    mean_value = df[column_name].mean()
    std_dev = df[column_name].std()
    lower_bound = mean_value - threshold * std_dev
    upper_bound = mean_value + threshold * std_dev
    new_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return new_df


def remove_all_outliers(df, not_count=None, threshold = OUTLIER_THRESHOLD):
    new_df = df.copy()
    for column_name in df.columns:
        if (is_numeric_dtype(new_df[column_name]) and (column_name not in not_count)):
            new_df = remove_outlier(df=new_df, column_name=column_name, threshold=OUTLIER_THRESHOLD)
    return new_df

def remove_few_entries(df, column, threshold=MINIMUM_ENTRIES_ACCEPTED):

    entry_counts = df[column].value_counts()
    valid_values = entry_counts[entry_counts >= MINIMUM_ENTRIES_ACCEPTED].index.tolist()
    return df[df[column].isin(valid_values)]


def convert_cat_to_num(df):

    df_new = df.copy()
    
    for col in df_new.columns:

        if df_new[col].dtype == 'object':
            cat_codes = pd.Categorical(df_new[col]).codes
            df_new[col] = cat_codes
    
    return df_new
