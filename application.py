# application.py

# ======================================================================================================================
# Imports
# ======================================================================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset, MonthEnd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

from utils import *
from prompts import *




# ======================================================================================================================
# Global Application Variables
# ======================================================================================================================

# ------------------------------------------------------------------------------
ml_options = [
	'Support Vector Machine',
	'AdaBoost: Decision Tree',
	# 'Neural Net: Sequential 3-2'
]

month_options = [
	'1', '2', '3'
]

# ------------------------------------------------------------------------------
svm_filename_defaults = [
	'svm_predictions_1m.pkl',
	'svm_predictions_2m.pkl',
	'svm_predictions_3m.pkl',
]
ada_filename_defaults = [
	'ada_predictions_1m.pkl',
	'ada_predictions_2m.pkl',
	'ada_predictions_3m.pkl',
]
nn_filename_defaults = [
	'nn_predictions_1m.pkl',
	'nn_predictions_2m.pkl',
	'nn_predictions_3m.pkl',
]

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

global monthly_predictions_list
monthly_predictions_list = []




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
def predict_future(ml_choice, monthly_df, current_date=pd.Timestamp('2020-01-31')):
	
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

	# generate predictions based on the machine learning model chosen
	if ml_choice == ml_options[0]:		# Support Vector Machine

		debug_print(f'-------- Support Vector Machine -- BEGIN')

		# create our model, fit, and predict the future (next n_months_predict)
		svm_model = svm.SVC()
		svm_model = svm_model.fit(X_train_scaled, y_train)
		y_future_prediction = svm_model.predict(X_test_scaled)

		debug_print(f'-------- Support Vector Machine -- END')

	elif ml_choice == ml_options[1]:	# AdaBoost: Decision Tree

		debug_print(f'-------- AdaBoost: Decision Tree -- BEGIN')

		# create our model, fit, and predict the future (next n_months_predict)
		ada_model = AdaBoostClassifier()
		ada_model = ada_model.fit(X_train_scaled, y_train)
		y_future_prediction = ada_model.predict(X_test_scaled)
		
		debug_print(f'-------- AdaBoost: Decision Tree -- END')

	elif ml_choice == ml_options[2]:	# Neural Net: Sequential 3-2

		debug_print(f'-------- Neural Net: Sequential -- BEGIN')

		# create our model, fit, and predict the future (next n_months_predict)

		# Define the the number of inputs (features) to the model
		number_input_features = 3
		# Define the number of neurons in the output layer
		number_output_neurons = 1
		# Define the number of hidden nodes for the first hidden layer
		hidden_nodes_layer1 =  (number_input_features + 1) // 2

		# Create the Sequential model instance
		nn = Sequential()
		# Add the first hidden layer
		nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation='relu'))
		# Add the output layer to the model specifying the number of output neurons and activation function
		nn.add(Dense(units=number_output_neurons, activation='linear'))
		# Display the Sequential model summary
		debug_print(f'-------- Neural Net: Summary\n{nn.summary()}')

		# create our model, fit, and predict the future (next n_months_predict)
		nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# Fit the model using 20 epochs and the training data
		fit_model = nn.fit(X_train_scaled, y_train, epochs=50)
		# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
		# model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)

		# Display the model loss and accuracy results
		debug_print(f"-------- Neural Net: Loss: {model_loss}, Accuracy: {model_accuracy}")

		y_future_prediction = (nn.predict(X_test_scaled)).astype("int32")

		debug_print(f'-------- Neural Net: Sequential -- END')

	else:
		debug_print(f'-------- NO MACHINE LEARNING MODEL SELECTED ----')
		debug_print(f'--------     HOW DID WE EVEN GET HERE??     ----')

	debug_print(f'-------- y_future_prediction: {y_future_prediction}')

	return y_future_prediction

# ----------------------------------------------------------------------------------------------------------------------
# this function takes the y_future_predictions and calculates months_to_invest
# returns months_to_invest
def calculate_investment_from_prediction(month_choice, y_future_prediction):
	
	# determine how many months to invest
	debug_print(f'---- calculate_investment_from_prediction()')
	debug_print(f'-------- month_choice       : {month_choice}')
	debug_print(f'-------- y_future_prediction: {y_future_prediction}')
	
	# initialize our investment allocation
	months_to_invest = 0
	
	# calculate our investment allocation based on the ML predictions
	for prediction in y_future_prediction:
		if prediction > 0:
			# for each positive return predicted, increase our investment allocation
			months_to_invest += 1
			# break when we've reached our chosen number of months to predict
			if months_to_invest >= month_choice:
				break
		else:
			# break at our first negative prediction
			break
	
	debug_print(f'-------- months_to_invest: {months_to_invest}')
	
	return months_to_invest

# ----------------------------------------------------------------------------------------------------------------------
# save our future predictions so we can graph them later
def save_monthly_predictions(current_date, current_close, current_return, current_target, y_future_prediction, months_to_invest):
	
	debug_print(f'---- save_monthly_predictions()')
	debug_print(f'-------- current_date -------: {current_date}')
	debug_print(f'-------- current_close ------: {current_close}')
	debug_print(f'-------- current_return -----: {current_return}')
	debug_print(f'-------- current_target -----: {current_target}')
	debug_print(f'-------- y_future_prediction : {y_future_prediction}')
	debug_print(f'-------- months_to_invest -- : {months_to_invest}')
	
	# ----------------------------------------------------------------------------------------------
	# save the data Toni needs to calculate the portfolio returns
	# ----------------------------------------------------------------------------------------------
	
	new_portfolio_return = {
		'Date':[current_date],
		'Close':[current_close],
		'Return':[current_return],
		'Months_To_Invest':[months_to_invest],
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
	
	# global monthly_predictions_col_list
	# global monthly_predictions_list

	# # create a datafrom to hold this months predictions
	# new_predictions_df = pd.DataFrame(columns=monthly_predictions_col_list)
	
	# new_predictions_df.Date[0] =current_date
	# new_predictions_df.Close[0]=current_close
	
	# new_predictions_df.Date[1] =current_date + MonthEnd(1)
	# new_predictions_df.Close[1]=abs(current_return) * y_future_prediction[0] + new_predictions_df.Close[0]
	
	# new_predictions_df.Date[2] =current_date + MonthEnd(2)
	# new_predictions_df.Close[2]=abs(current_return) * y_future_prediction[1] + new_predictions_df.Close[1]
	
	# new_predictions_df.Date[3] =current_date + MonthEnd(3)
	# new_predictions_df.Close[3]=abs(current_return) * y_future_prediction[2] + new_predictions_df.Close[2]
	
	# monthly_predictions_list.append(new_predictions_df)

	# # print the last 5 monthly predictions
	# debug_print(monthly_predictions_list[len(monthly_predictions_list)-3:])
	
	return None

# ----------------------------------------------------------------------------------------------------------------------
# TODO: how we plot and save each set of predictions over time
# ----------------------------------------------------------------------------------------------------------------------
def plot_save_predictions(file_path, output_path):

	debug_print('---- plot_save_predictions()')

	# some print statements to aid in debugging
	debug_print(f'---- plot_save_predictions() -----------------------------------------------------------------------------------------')
	debug_print(f'-------- file_path --: {file_path}')
	debug_print(f'-------- output_path : {output_path}')

	# global monthly_predictions_list
	
	# monthly_prediction_plots = []

	# for prediction_df in monthly_predictions_list:
	# 	# hello
	
	# df = load_pickle(file_path)
	# fig = df.cumsum().plot()
	# save_image(fig, output_path)

	return None

# ----------------------------------------------------------------------------------------------------------------------
def main():

	debug_print('-----------------------------------------------------------------------------------------------------')
	debug_print('---- main() -----------------------------------------------------------------------------------------')
	debug_print('-----------------------------------------------------------------------------------------------------')
	
	continue_execution = True

	# ----------------------------------------------------------------------------------------------
	while continue_execution:
		
		'''
		1. display the welcome message
		2. ask the user which machine learning model to use
		3. ask the user how many months to predict
		4. ask the user to specify the filename to save the portfolio_predictions for later use
		5. ask the user if they want to continue
		'''

		# print a welcome message for the user
		welcome_message()

		# get the user's stock choice
		ml_choice = prompt_multiple_choice(
			"Please select the Machine Learning model you wish to use:",
			ml_options
		)
		debug_print(f'-- Your ML choice was: {ml_choice}')

		month_choice = int(prompt_multiple_choice(
			"Please select the number of months to predict:",
			month_options
		))
		debug_print(f'-- Your months to predict was: {month_choice}')

		debug_print('-----------------------------------------------------------------------------------------------------')
		debug_print('-----------------------------------------------------------------------------------------------------')	

		# ----------------------------------------------------------------------------------------------
		# load the VTI total market ETF and resample the daily data to monthly
		total_market_df = resample_dataframe(load_csv('VTI'))
		debug_print('-- loaded & resampled VTI.csv')
		
		# prep VTI for Machine Learning
		total_market_df = prep_data_for_ML(total_market_df)
		debug_print('-- prepped VTI for ML')

		debug_print('-----------------------------------------------------------------------------------------------------')
		debug_print('-- calculating our future predictions ---------------------------------------------------------------')
		debug_print('-----------------------------------------------------------------------------------------------------')
	
		# ----------------------------------------------------------------------------------------------
		# skip some months at the beginning so we have enough training data
		months_to_skip = 12
		debug_print(f'-- months_to_skip: {months_to_skip}')
		# run for this many months total
		# months_to_run  = int(((total_market_df.index.max() - total_market_df.index.min()) / np.timedelta64(1,'M')) / 2)
		# ^^^ this calculates half the time
		months_to_run  = int(((total_market_df.index.max() - total_market_df.index.min()) / np.timedelta64(1,'M'))) + 1
		# ^^^ this runs for basically the whole time
		debug_print(f'-- months_to_run:  {months_to_run}')

		# notify the user it's time to start calculating the portfolio predictions
		prompt_confirm('Ready to calculate future predictions. Press [ENTER] to continue')

		# ----------------------------------------------------------------------------------------------
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
			y_future_prediction = predict_future(ml_choice, total_market_df, current_date)
			debug_print('-- future predicted')
			
			months_to_invest = calculate_investment_from_prediction(month_choice, y_future_prediction)

			# ----------------------------------------------------------------------------------------------
			# save our current months predictions for later plotting
			save_monthly_predictions(
				current_date,
				total_market_df.loc[current_date]['Close'],
				total_market_df.loc[current_date]['Current Return'],
				total_market_df.loc[current_date]['Current Target'],
				y_future_prediction,
				months_to_invest
			)
			debug_print('-- predictions saved')
			debug_print('-----------------------------------------------------------------------------------------------------')

			# NOTE: in teh future we can re-work Toni's code to update the portfolio on a monthly basis with a function like...
			# update_portfolio(current_date, this_months_return, months_to_invest)
			#  this function should...
			#   - update the portfolio returns each month
			#   - invest-and-rebalance at the same time if months_to_invest is positive

		debug_print('-----------------------------------------------------------------------------------------------------')
		debug_print('-- calculated all future predictions ---')
		debug_print('-----------------------------------------------------------------------------------------------------')

		# notify the user the portfolio prediction calculates are complete
		prompt_confirm('All future predictions are calculated. Press [ENTER] to continue')

		# ----------------------------------------------------------------------------------------------
		# get the default filename based on the ml_choice and month_choice
		default_filename = 'file.pkl'
		if ml_choice == ml_options[0]:		# Support Vector Machine
			default_filename = svm_filename_defaults[month_choice-1]
		elif ml_choice == ml_options[1]:	# AdaBoost: Decision Tree
			default_filename = ada_filename_defaults[month_choice-1]
		elif ml_choice == ml_options[2]:	# Neural Net: Sequential 3-2
			default_filename = nn_filename_defaults[month_choice-1]
		debug_print(f'-- default_filename: {default_filename}')

		# ----------------------------------------------------------------------------------------------
		# prompt the user to enter the filepath in which to save the portfolio predictions
		prediction_filename = prompt_file_path(
			"Please specify the filename of your portfolio predictions:",
			default=default_filename
		)
		debug_print(f'-- The filename entered was: {prediction_filename}')

		# ----------------------------------------------------------------------------------------------
		# save our portfolio predictions
		global portfolio_returns_pred_df
		debug_print('-- save [portfolio_returns_pred_df] to pickle -------------------------------------------------------')
		debug_print(portfolio_returns_pred_df)
		save_pickle(portfolio_returns_pred_df, './resources/predictions/portfolio_' + prediction_filename)

		# ----------------------------------------------------------------------------------------------
		# plot our predictions
		global predictions_to_plot_df
		debug_print('-- save [predictions_to_plot_df] to pickle ----------------------------------------------------------')
		debug_print(predictions_to_plot_df)
		save_pickle(predictions_to_plot_df, './resources/predictions/plot_me_' + prediction_filename)
		
		plot_predictions_file = './resources/predictions/plot_' + prediction_filename
		output_path = './resources/images/plot_of_' + prediction_filename
		plot_save_predictions(plot_predictions_file, output_path)

		# # ----------------------------------------------------------------------------------------------
		# # plot our predictions a different way
		# debug_print(monthly_predictions_list[len(monthly_predictions_list)-5:])
		global monthly_predictions_list
		save_pickle(monthly_predictions_list, './resources/predictions/plot_me_' + prediction_filename)


		# ----------------------------------------------------------------------------------------------
		continue_execution = prompt_confirm("Do you want to continue", qmark='?')
		debug_print(f'-- continue_execution: {continue_execution}')
		
	return None




# ======================================================================================================================
if __name__ == "__main__":
	main()
	
	
	# add some print statements to make sure the output is good
	
# ======================================================================================================================
