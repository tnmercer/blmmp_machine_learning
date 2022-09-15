from utils import *
import questionary as qt

def welcome_message():

	print()
	print('Welcome to MMP: Retirement Prediction')
	print()
	print("This Monthly Money Planner application helps you plan your retirement portoflio investment strategy. This application lets you choose one of three machine learning models for prediction . Results are displayed in an interactive graph that you man inspect. Enjoy!")
	print()

	return None

def prompt_multiple_choice(question_text, question_options):
	'''This function wraps the Questionary .select() function used to prompt the user for an answer.
	'''

	debug_print('---- prompt_multiple_choice()')

	result = qt.select(
		question_text,
		choices = question_options
	).ask()
	print()

	return result

def prompt_single_choice(question_text):

	debug_print('---- prompt_single_choice()')
	
	result = qt.confirm(question_text).ask()
	print()

	return result

def prompt_text_input(question_text):

	debug_print('---- prompt_text_input()')

	result = qt.text(question_text).ask()
	print()

	return result

def prompt_file_path(question_text, default='file.pkl'):

	debug_print('---- prompt_file_path()')

	result = qt.path(question_text, default).ask()
	print()

	return result

def prompt_confirm(question_text, qmark=':'):

	debug_print('---- prompt_confirm()')

	result = qt.confirm(question_text, qmark=qmark).ask()
	print()

	return result
