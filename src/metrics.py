import pandas as pd
import numpy as np

def stats_on_column(column_name, df):

    column_data = df[column_name]
    if np.isnan(column_data).all():
        return pd.DataFrame(index=['mean', 'median', 'min', 'max', '25%', '75%', 'range', 'IQR'], columns=[column_name])
    else:
        stats = {
            'mean': column_data.mean(),
            'median': column_data.median(),
            'min': column_data.min(),
            'max': column_data.max(),
            '25%': column_data.quantile(0.25),
            '75%': column_data.quantile(0.75),
            'range': column_data.max() - column_data.min(),
            'IQR': column_data.quantile(0.75) - column_data.quantile(0.25)
        }
        return pd.DataFrame(stats, index=[column_name])


def get_stats_on_df(column_name, index_column_name, df):
    """
    At this point, we use the former method in a recursion throught the whole dataframe and replace indexing 
    in a final dataframe that has as index the unique values of the column reffered as index_column_name
    """

    unique_index_values = df[index_column_name].unique()
    final_df = pd.DataFrame()
    for index_value in unique_index_values:
        index_group_df = df[df[index_column_name] == index_value]
        index_group_stats = stats_on_column(column_name, index_group_df)
        index_group_stats.index.name = index_column_name
        index_group_stats.index = [index_value]
        final_df = pd.concat([final_df, index_group_stats], axis=0)
    return final_df




def competitiveness_index(df, weights):

    """
    Implementing the competitiveness index as described in Noteboook Part2.ipynb
    """

    df_copy = df.copy()

    df_copy['Floor_Factor'] = np.where(df_copy['floor'] >= 3, 3, df_copy['floor'])
    df_copy['Floor_Factor'] = df_copy['Floor_Factor'] * -1
    df_copy['Rooms_Factor'] = (1 + df_copy['rooms'])**1.5
    df_copy['Deviation_Factor'] = np.sqrt((df_copy['price_per_sqrm'] - df_copy.groupby('geography_name')['price_per_sqrm'].transform('median'))**2)
    df_copy['Deviation_Factor'] = df_copy.groupby('geography_name')['Deviation_Factor'].transform(lambda x: x / x.max())
    ad_type_dict = {'simple': 0, 'up': 1, 'star': 2, 'premium': 3}

    df_copy['Ad_Type_Factor'] = df_copy['ad_type'].map(ad_type_dict)

    df_copy['Year_of_Construction'] = pd.cut(df_copy['year_of_construction'], bins=[-np.inf, 1970, 2000, np.inf], labels=['old', 'mid', 'new'])
    construction_dict = {'old': 2, 'mid': 1, 'new': -2}
    df_copy['Construction_Factor'] = df_copy['Year_of_Construction'].map(construction_dict)

    df_copy['Competitiveness_Index'] = np.dot(df_copy[['Floor_Factor', 'Rooms_Factor', 'Deviation_Factor', 'Ad_Type_Factor', 'Construction_Factor']], weights)

    return df_copy


def calculate_correlations(df,target='price'):

    # Convert categorical columns to numeric values using one-hot encoding
    df_numeric = pd.get_dummies(df)

    # Calculate correlation values between 'target' (i.e., 'price') and all other columns in the dataframe
    corr_df = df_numeric.corr()[target].reset_index()
    corr_df.columns = ['Column', 'Correlation']

    return corr_df.set_index('Column')
