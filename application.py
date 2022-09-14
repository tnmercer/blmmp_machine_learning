# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset, MonthEnd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import svm


from utils import *
# from plot_function import *


# ======================================================================================================================
# Global Application Variables
# ======================================================================================================================

# ------------------------------------------------------------------------------
portfolio_returns_pred_df_col_list = [
	'Date',
	'Close',
	'Return',
	'Months_To_Invest',
]

global portfolio_returns_pred_df
portfolio_returns_pred_df = None
portfolio_returns_pred_df = pd.DataFrame(columns=portfolio_returns_pred_df_col_list)
portfolio_returns_pred_df.set_index('Date', inplace=True)

# ------------------------------------------------------------------------------
predictions_to_plot_df_col_list = [
	'Date',
	'Current Target',
	'Future Target',
]

global predictions_to_plot_df
predictions_to_plot_df = None
predictions_to_plot_df = pd.DataFrame(columns=predictions_to_plot_df_col_list)
predictions_to_plot_df.set_index('Date', inplace=True)

# ------------------------------------------------------------------------------
monthly_predictions_col_list = [
	'Date',
	'Close',
]

global monthly_predictions_plot_list
monthly_predictions_plot_list = []



# ======================================================================================================================
# Core Application Functions
# ======================================================================================================================



# ----------------------------------------------------------------------------------------------------------------------
# this function loads the price data for a particular ticker
# it returns a dataframe with a Datetime index columns, and 
# Open, High, Low, Closed, and Volume columns
def load_csv(ticker):
	
	debug_print('---- load_csv()')
	debug_print(f'-------- ticker: {ticker}')
	
	# load the ticker data
	# df = load_data(ticker, set_index=True, index_column='Date')
	df = load_data(ticker)
	
	# check if the first date index is less than 2012-01-01
	if df.index[0] < pd.Timestamp('2012-01-01'):
		# if so, drop all date rows before January 01, 2012
		# we're performing this step so all stock data starts
		# at the same time
		df = df[df.index > pd.Timestamp('2012-01-01')]
		
		# then check if 'Adj Close' and 'Ret_Index' columms exist
		# if so, remove them from the dataframe
		if {'Adj Close'}.issubset(df.columns):
			df = df.drop(columns='Adj Close')
		if {'Ret_Index'}.issubset(df.columns):
			df = df.drop(columns='Ret_Index')
		
		# then save the modified dataframe back to CSV
		save_data(df, filename=ticker+'.csv')
	
	# now return the modified dataframe
	return df

# ----------------------------------------------------------------------------------------------------------------------
def resample_dataframe(df):
	
	debug_print(f'---- resample_dataframe()')
	debug_print(f'-------- df:\n{df}')
	
	df_to_resample = df.copy()
	# drop 'Adj Close' and 'Ret_Index' columms if they exist
	if {'Adj Close'}.issubset(df_to_resample.columns):
		df_to_resample = df_to_resample.drop(columns='Adj Close')
	if {'Ret_Index'}.issubset(df_to_resample.columns):
		df_to_resample = df_to_resample.drop(columns='Ret_Index')
	
	how_to_resample = {
        'Open':'first',
        'High':'max',
        'Low':'min',
        'Close':'last',
        'Volume':'sum'
    }
	
	resampled_df = None
	resampled_df = df_to_resample.resample('1M').agg(how_to_resample)
	
	return resampled_df

# ----------------------------------------------------------------------------------------------------------------------
# this function prepares the stock data for machine learning
# it generates 
def prep_data_for_ML(df):
	
	debug_print(f'---- prep_data_for_ML()')
	debug_print(f'-------- df:\n{df}')
	
	df['Current Return'] = df['Close'].pct_change()
	df['Future Returns'] = df['Close'].pct_change().shift(-3)
	
	df['Current Target']= 0.0
	df.loc[(df['Current Return'] >= 0), 'Current Target'] = 1
	df.loc[(df['Current Return'] < 0), 'Current Target'] = -1
	
	df['Future Target'] = 0.0
	df.loc[(df['Future Returns'] >= 0), 'Future Target'] = 1
	df.loc[(df['Future Returns'] < 0), 'Future Target'] = -1
	
	df = df.dropna()

	debug_print(f'-------- return df:\n{df}')
	
	return df

# ----------------------------------------------------------------------------------------------------------------------
def predict_future(monthly_df, current_date=pd.Timestamp('2020-01-31')):
	
	debug_print(f'---- predict_future()')
	debug_print(f'-------- current_date: {current_date}')
	debug_print(f'-------- monthly_df:\n{monthly_df.tail()}')
	
	# TODO: moving this code into an initialization function
	# daily_df = load_csv(ticker)
	# monthly_df =  resample_dataframe(daily_df)
	
	# commenting this code because we should provide a current_date each time
	# this function is run
	# current_date = monthly_df.index.max() - DateOffset(months=12)

	training_begin = None
	training_end = None
	test_begin = None
	test_end = None
	
	# calculating the beginning and end of our train and test time frames
	training_begin = monthly_df.index.min()		# training starts at the beginning
	training_end = current_date - MonthEnd(3)	# training ends 3 months back
	test_begin = current_date - MonthEnd(2)		# test includes 3 monhts total
	test_end   = current_date

	debug_print(f'-------- train begin: {training_begin} -------------------------------------------')
	debug_print(f'-------- train end  : {training_end} -------------------------------------------')
	debug_print(f'-------- test begin : {test_begin} -------------------------------------------')
	debug_print(f'-------- test end   : {test_end} -------------------------------------------')

	# creating our X (features) and y (prediction) dataframes
	X = monthly_df.drop(columns=[
		'Open','High', 'Low',
		'Current Return','Current Target',
		'Future Returns','Future Target'
		])
	y = monthly_df['Future Target']
	
	# creating our training and testing sub-sets
	X_train = X.loc[training_begin:training_end]
	y_train = y.loc[training_begin:training_end]
	X_test = X.loc[test_begin:test_end]
	y_test = y.loc[test_begin:test_end]

	debug_print(f'-------- X_train:\n{X_train.tail()}')
	debug_print(f'-------- y_train:\n{y_train[-5:]}')
	debug_print(f'-------- X_test:\n{X_test.tail()}')
	debug_print(f'-------- y_test:\n{y_test[-5:]}')

	# scale the features we provide to our model
	scaler = StandardScaler()
	X_scaler = scaler.fit(X_train)
	X_train_scaled = X_scaler.transform(X_train)
	X_test_scaled = X_scaler.transform(X_test)

	# create our model, fit, and predict the future (next 3 months)
	svm_model = svm.SVC()
	svm_model = svm_model.fit(X_train_scaled, y_train)
	y_future_prediction = svm_model.predict(X_test_scaled)

	debug_print(f'-------- y_future_prediction: {y_future_prediction}')

	return y_future_prediction

# ----------------------------------------------------------------------------------------------------------------------
# this function takes the y_future_predictions and calculates months_to_invest
# returns months_to_invest
def calculate_investment_from_prediction(y_future_prediction):
	
	debug_print(f'---- calculate_investment_from_prediction()')
	debug_print(f'-------- y_future_prediction: {y_future_prediction}')
	
	# determine how many months to invest
	months_to_invest = 0
	
	# for each prediction, add a month's worth of investing 
	for prediction in y_future_prediction:
		if prediction > 0:
			months_to_invest += 1
		else:
			break
	
	debug_print(f'-------- months_to_invest: {months_to_invest}')
	
	return months_to_invest

# ----------------------------------------------------------------------------------------------------------------------
# save our future predictions so we can graph them later
def save_monthly_predictions(current_date, current_close, current_return, current_target, y_future_prediction):
	
	debug_print(f'---- save_monthly_predictions()')
	debug_print(f'-------- current_date: {current_date}')
	debug_print(f'-------- current_close: {current_close}')
	debug_print(f'-------- current_return: {current_return}')
	debug_print(f'-------- current_target: {current_target}')
	debug_print(f'-------- y_future_prediction: {y_future_prediction}')
	
	# ----------------------------------------------------------------------------------------------
	# save the data Toni needs to calculate the portfolio returns
	# ----------------------------------------------------------------------------------------------
	
	new_portfolio_return = {
		'Date':[current_date],
		'Close':[current_close],
		'Return':[current_return],
		'Months_To_Invest':[calculate_investment_from_prediction(y_future_prediction)],
	}
	new_portfolio_return_df = pd.DataFrame.from_dict(new_portfolio_return)
	new_portfolio_return_df.set_index('Date', inplace=True)
	
	debug_print('-------- new_portfolio_return_df')
	debug_print(new_portfolio_return_df)

	global portfolio_returns_pred_df
	portfolio_returns_pred_df = pd.concat([portfolio_returns_pred_df, new_portfolio_return_df])
	
	debug_print('-------- portfolio_returns_pred_df')
	debug_print(portfolio_returns_pred_df.tail(5))
	
	
	# ----------------------------------------------------------------------------------------------
	# save the predictions we need to plot
	# ----------------------------------------------------------------------------------------------
	
	# columns = 'Date', 'Current Return', 'Future Returns',
	
	new_predictions = {
		'Date'          : [current_date],
		'Current Target': [current_target],
		'Future Target' : [y_future_prediction[0]],
	}
	new_predictions_df = pd.DataFrame.from_dict(new_predictions)
	new_predictions_df.set_index('Date', inplace=True)

	debug_print('-------- new_predictions_df')
	debug_print(new_predictions_df)
	
	global predictions_to_plot_df
	predictions_to_plot_df = pd.concat([predictions_to_plot_df, new_predictions_df])
	
	debug_print('-------- predictions_to_plot_df')
	debug_print(predictions_to_plot_df.tail(5))
	
	
	# ----------------------------------------------------------------------------------------------
	# TODO: a more complicated way to plot our predictions
	# ----------------------------------------------------------------------------------------------
	
	# columns = 'Date', 'Close'
	
	# monthly_predictions_col_list
	
	# # create a datafrom to hold this months predictions
	# new_predictions_df = pd.DataFrame(columns=monthly_predictions_col_list)
	
	# new_predictions_df.Date[0] =current_date
	# new_predictions_df.Close[0]=current_close
	
	# new_predictions_df.Date[1] =current_date + DateOffset(1)
	# new_predictions_df.Close[1]=abs(current_return) * y_future_prediction[0] + new_predictions_df.Close[0]
	
	# new_predictions_df.Date[2] =current_date + DateOffset(2)
	# new_predictions_df.Close[2]=abs(current_return) * y_future_prediction[1] + new_predictions_df.Close[1]
	
	# new_predictions_df.Date[3] =current_date + DateOffset(3)
	# new_predictions_df.Close[3]=abs(current_return) * y_future_prediction[2] + new_predictions_df.Close[2]
	
	# monthly_predictions_plot_list.append(new_predictions_df)
	
	# print the last 5 monthly predictions
	# debug_print(monthly_predictions_plot_list[len(monthly_predictions_plot_list)-5:])
	
	
	return None

# ----------------------------------------------------------------------------------------------------------------------
def main():
	debug_print('-----------------------------------------------------------------------------------------------------')
	debug_print('---- main() -----------------------------------------------------------------------------------------')
	debug_print('-----------------------------------------------------------------------------------------------------')
	
	# load the VTI total market ETF and resample the daily data to monthly
	total_market_df = resample_dataframe(load_csv('VTI'))
	debug_print('-- loaded & resampled VTI.csv')
	
	# prep VTI for Machine Learning
	total_market_df = prep_data_for_ML(total_market_df)
	debug_print('-- prepped VTI for ML')

	# save current_date (to remember where we are in time)
	# save current_date_idx = 12
	# make current_date = total_market_df.index[12]

	# we need the last_investment_date so we know how many months to invest
	# this needs to be moved into Toni's code
	global last_investment_date
	last_investment_date = None

	debug_print('-----------------------------------------------------------------------------------------------------')
	debug_print('-- calculating our future predictions ---------------------------------------------------------------')
	debug_print('-----------------------------------------------------------------------------------------------------')
	
	# skip some months at the beginning so we have enough training data
	months_to_skip = 12
	debug_print(f'-- months_to_skip: {months_to_skip}')
	# run for this many months total
	# months_to_run  = int(((total_market_df.index.max() - total_market_df.index.min()) / np.timedelta64(1,'M')) / 2)
	# ^^^ this calculates half the time
	months_to_run  = int(((total_market_df.index.max() - total_market_df.index.min()) / np.timedelta64(1,'M'))) + 1
	# ^^^ this runs for basically the whole time
	debug_print(f'-- months_to_run:  {months_to_run}')

	# for each month, calculate our future predictions and invest if appropriate
	for current_date in total_market_df.index:
		
		debug_print('-----------------------------------------------------------------------------------------------------')
		debug_print(f'-- current date: {current_date}')
		
		# skip the first 12 months
		if current_date < total_market_df.index[months_to_skip]:
			debug_print('-- skipped')
			continue
		
		# ----------------------------------------------------------------------
		# stop running after N months for test purposes
		# calculate total 
		if current_date > total_market_df.index[months_to_run]:
			debug_print('-- stopped')
			break
		# ----------------------------------------------------------------------

		# get our future predictions
		y_future_prediction = predict_future(total_market_df, current_date)
		debug_print('-- future predicted')

		# save our current months predictions for later plotting
		# NOTE: save_monthly_predictions function is INCOMPLETE, see above
		save_monthly_predictions(
			current_date,
			total_market_df.loc[current_date]['Close'],
			total_market_df.loc[current_date]['Current Return'],
			total_market_df.loc[current_date]['Current Target'],
			y_future_prediction
		)
		debug_print('-- predictions saved')
		debug_print('-----------------------------------------------------------------------------------------------------')

		# FUTURE re-work Toni's code to update the portfolio on a monthly basis
		# update_portfolio(current_date, this_months_return, months_to_invest)
		#  this function should...
		#   - update the portfolio returns each month
		#   - invest-and-rebalance at the same time if months_to_invest is positive
	
	debug_print('-----------------------------------------------------------------------------------------------------')
	debug_print('-- calculated all future predictions ---')
	debug_print('-----------------------------------------------------------------------------------------------------')

	global portfolio_returns_pred_df
	debug_print('-- save [portfolio_returns_pred_df] to pickle -------------------------------------------------------')
	debug_print(portfolio_returns_pred_df)
	save_pickle(portfolio_returns_pred_df, './Resources/portfolio_returns_pred_df.pkl')

	global predictions_to_plot_df
	debug_print('-- save [predictions_to_plot_df] to pickle ----------------------------------------------------------')
	debug_print(predictions_to_plot_df)
	save_pickle(predictions_to_plot_df, './Resources/predictions_to_plot_df.pkl')
	
	# then Dan and Toni can use this with your respective code
	
	return None




# ======================================================================================================================
if __name__ == "__main__":
	main()
	
	
	# add some print statements to make sure the output is good
	
# ======================================================================================================================
