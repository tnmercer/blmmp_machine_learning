import datetime as dt
import pandas as pd
import pandas_datareader as pdr
from pathlib import Path

from config import *



# ==================================================================================================
# global variables
# ==================================================================================================

''' moved to config.py
DEBUG = True
'''



# ==================================================================================================
# core functions
# ==================================================================================================
def initialize_df(column_names, set_index=False, index_column=''):

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

	# stock_tickers = ['SPY','QQQ','AAPL','AMZN','GOOG','META','MSFT']
	# stock_df = {}
	# for ticker in stock_tickers :
	# 	# download the stock data for this ticker
	# 	stock_df[ticker] = pdr.get_data_yahoo(ticker, start_date, end_date, ret_index=True)
	# 	# reset the index so 'Date' becomes a column again
	# 	stock_df[ticker].reset_index(inplace=True)

	stock = pdr.get_data_yahoo(ticker, start_date, end_date, ret_index=False)

	if DEBUG:
		print(f'download_data -- ticker: {ticker}')
		print(stock.head(20))
	
	#  stock.to_csv(Path('data/'+ticker+'.csv'))

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

	path = Path(dir+'/'+filename)
	result = df.to_csv(path)

	if DEBUG:
		print(f'save_data ------ saving {path}')

	return None

# --------------------------------------------------------------------------------------------------
def load_data(ticker, set_index=False, index_column=''):
	
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

	# generate a path for the ticker data we want to load
	path = Path('data/'+ticker+'.csv')
	
	# check for the existence of that file
	if not path.is_file():
		if DEBUG:
			print(f'load_data ------ path {path} doesn\'t exist.')
			print(f'load_data ------ initiating download for [{ticker}] now...')
		# since data for that file doesn't exist, let's download it
		stock_data_to_save = download_stock_data(ticker)
		# and then save it out to disk so we have it for next time
		save_data(stock_data_to_save, filename=ticker+'.csv')

	data_df = pd.read_csv(
		path, 
	#	index_col='Date', 
		parse_dates=True,
		infer_datetime_format=True
	)
	if (set_index and index_column==''):
		assert(f'load_data() -> index_column is blank, please provide the correct column name')
	if (set_index):
		data_df.set_index(index_column,inplace=True)

	if DEBUG:
		print('---- symbol_df ----')
		print(data_df.head())
		print(data_df.tail())

	return data_df
