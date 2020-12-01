import pandas as pd
import math
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

filename = 'all-states-history.csv'
valuable_rows = ['state', 'date', 'death', 'deathConfirmed', 'hospitalizedCurrently', 'positiveCasesViral', 
                'positiveIncrease', 'totalTestsPeopleViral', 'totalTestsPeopleViralIncrease', 'totalTestsViral']
target_ranges = []
df = pd.read_csv(filename)
nf = pd.DataFrame(df)
nf.drop(nf.columns.difference(valuable_rows), 1, inplace = True) # na_values='0'
nfrows = nf.shape[0]
print(nf.head())