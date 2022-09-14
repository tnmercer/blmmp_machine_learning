import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt


def results_comparison(y_future_prediction, original_df):

    # create a function that creates an original dataframe
    # create a dataframe with new data
    # update the original dataframe with the new data
    
    new_df = y_future_prediction
    updated_df = pd.concat([original_df,new_df]).drop_duplicates(['Date'],keep='last').sort_values('Date')
    original_df = updated_df
    
    return original_df
    return original_df.plot(x='Date')