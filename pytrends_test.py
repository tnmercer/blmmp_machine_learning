from ast import keyword
import requests
import pandas as pd
from pytrends.request import TrendReq

pytrend = TrendReq(hl='en-US', tz=360)


# Google suggestions around search term
# kw_list = pytrend.suggestions(keyword='Passive')

# Dataframe of search suggestions
# df = pd.DataFrame(kw_list)

# print(df.head())


# Create payload
pytrend.build_payload(kw_list=['market crash'])

# Interest over time
interest_over_time_df = pytrend.interest_over_time()

print(interest_over_time_df.head())
