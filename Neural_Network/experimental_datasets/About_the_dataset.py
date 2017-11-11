import pandas as pd
from pandas.tools.plotting import scatter_matrix

from matplotlib import pyplot as plt


df = pd.read_csv('Data/fluMl.csv')
print df.shape
df = df.dropna()
#print df.Flu.value_counts()
#print df.shape
#print df.describe()
#plt.figure(figsize=(50,50))
#df.plot(kind='box', subplots=True, layout=(2,10), sharex=False, sharey=False)
df.hist(figsize=(50,50))
#scatter_matrix(df, figsize=(50,50))
plt.show()
