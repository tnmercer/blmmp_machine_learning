import datetime as dt
import pandas as pd
import pandas_datareader as pdr
from pandas.tseries.offsets import BDay
from datetime import date
from pathlib import Path
import pickle
import plotly.graph_objects as go
import numpy as np


from config import *

# ==================================================================================================
# core functions
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
def debug_print(s):
	if DEBUG:
		ret = print(s)
	return ret

# --------------------------------------------------------------------------------------------------
def initialize_df(column_names, set_index=False, index_column=''):
	
	debug_print('---- initialize_df()')
	
	df = pd.DataFrame(
		columns=column_names
	)
	if (set_index and index_column==''):
		assert(f'load_data() -> index_column is blank, please provide the correct column name')
	if (set_index):
		df.set_index(index_column,inplace=True)
	
	return df


# --------------------------------------------------------------------------------------------------
def download_stock_data(
	ticker='SPY',
	start_date=dt.datetime(2000,1,1),
	end_date=dt.datetime.now()):

	"""download_stock_data() TODO: a summary of what this function does 
	
	TODO: add a detailed description if necessary
	
	Parameters
	----------
	TODO: update the list of parameters
	
	...use this format when listing parameters...
	<variable_name> : <variable_type> (required/optional)
		<variable_description_or_purpose>
	
	...for example...
	df : pandas.DataFrame (required)
		A OHLC dataframe containing the pricing data related to this order.
	
	Returns
	-------
	TODO: specify the return value
	"""

	
	debug_print('---- download_stock_data()')
	
	stock = pdr.get_data_yahoo(ticker, start_date, end_date, ret_index=False)
	
	debug_print(f'download_data -- ticker: {ticker}')
	debug_print(stock.head(20))
	
	return stock


# --------------------------------------------------------------------------------------------------
def save_data(df, dir='data', filename='file.csv'):
	"""load_stock_data() TODO: a summary of what this function does 
	
	TODO: add a detailed description if necessary
	
	Parameters
	----------
	TODO: update the list of parameters
	
	...use this format when listing parameters...
	<variable_name> : <variable_type> (required/optional)
		<variable_description_or_purpose>
	
	...for example...
	df : pandas.DataFrame (required)
		A OHLC dataframe containing the pricing data related to this order.
	
	Returns
	-------
	TODO: specify the return value
	"""
	
	debug_print('---- save_data()')
	
	path = Path(dir+'/'+filename)
	result = df.to_csv(path)
	
	debug_print(f'save_data ------ saving {path}')
	
	return None


# --------------------------------------------------------------------------------------------------
# def load_data(ticker, set_index=False, index_column=''):
def load_data(ticker):
	
	"""load_data() TODO: a summary of what this function does 
	
	TODO: add a detailed description if necessary
	
	Parameters
	----------
	TODO: update the list of parameters
	
	...use this format when listing parameters...
	<variable_name> : <variable_type> (required/optional)
		<variable_description_or_purpose>
	
	...for example...
	df : pandas.DataFrame (required)
		A OHLC dataframe containing the pricing data related to this order.
	
	Returns
	-------
	TODO: specify the return value
	"""
	
	debug_print('---- load_data()')
	
	# generate a path for the ticker data we want to load
	path = Path('data/'+ticker+'.csv')
	
	# check for the existence of that file
	if not path.is_file():
		debug_print(f'load_data ------ path {path} doesn\'t exist.')
		debug_print(f'load_data ------ initiating download for [{ticker}] now...')
		# since data for that file doesn't exist, let's download it
		stock_data_to_save = download_stock_data(ticker)
		# and then save it out to disk so we have it for next time
		save_data(stock_data_to_save, filename=ticker+'.csv')

	data_df = pd.read_csv(
		path, 
		index_col='Date', 
		parse_dates=True,
		infer_datetime_format=True
	)
	# if (set_index and index_column==''):
	# 	assert(f'load_data() -> index_column is blank, please provide the correct column name')
	# if (set_index):
	# 	data_df.set_index(index_column,inplace=True)

	debug_print('---- symbol_df ----')
	debug_print(data_df.head())
	debug_print(data_df.tail())

	return data_df


# --------------------------------------------------------------------------------------------------
def create_portfolio():
	
	debug_print('---- create_portfolio()')
	
	portfolio = pd.DataFrame()
	
	for key in all_portfolios:
		for key in key:
			# Create dataframe from CSV file
			df = pd.read_csv(f"./data/{key}.csv")
			
			# Drop columns and set date index for concat
			df = df[['Date','Close']].set_index('Date')
			
			# Rename to Ticker
			df = df.rename(columns={'Close':key})
			
			# Concat to empty dataframe
			all_funds_df = pd.concat([all_funds_df, df], axis=1)
	
	# drop na values 
	all_funds_df = all_funds_df.dropna()
	
	debug_print(all_funds_df.head())
	
	# save to pkl file
	dataframe = all_funds_df
	output = open('data/all_funds_df.pkl', 'wb')
	pickle.dump(dataframe, output)
	output.close()
	
	return None

# --------------------------------------------------------------------------------------------------

def portfolio_return(selected_portfolio_weights):
    portfolio_returns = pd.DataFrame()

    for key in selected_portfolio_weights.keys():
        # Create dataframe from CSV file
        df = pd.read_csv(f"./data/{key}.csv")

        # Drop columns and set date index for concat
        df = df[['Date','Close']].set_index('Date')
        df.index = pd.to_datetime(df.index)

        # Rename columns to Ticker
        df = df.rename(columns={'Close':key})

        # Change Close to return
        df = df.pct_change()

        # Concat to empty dataframe
        portfolio_returns = pd.concat([portfolio_returns, df], axis=1)

    # drop na values 
    portfolio_returns = portfolio_returns.dropna()
    
    return portfolio_returns

# --------------------------------------------------------------------------------------------------

def credit_holding_acct(holding_account, ira_contribution_limit):
    
	holding_account = ira_contribution_limit + holding_account

	return holding_account

# --------------------------------------------------------------------------------------------------

def debit_holding_acct(holding_account, trans_amt):
    
	holding_account = holding_account - trans_amt

	return holding_account

# --------------------------------------------------------------------------------------------------
def save_pickle(df, path):
	
	debug_print('---- save_pickle()')
	
	output = open(path, 'wb')
	pickle.dump(df, output)
	output.close()
	
	return None


# --------------------------------------------------------------------------------------------------
def load_pickle(path):
	
	debug_print('---- load_pickle()')
	
	pkl_file = open(Path(path), 'rb')
	df = pickle.load(pkl_file)
	pkl_file.close()
        
	return df

# --------------------------------------------------------------------------------------------------

def save_image(fig, path):
	
	fig.write_image(path)

	return None
