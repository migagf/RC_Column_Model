# New figures 4 paper

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use latex for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the nondimentional parameter data
df = pd.read_csv('data_all.csv')

data_columns = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'FailureType']
data_col_labems = ['AR', 'LRR', 'SRR', 'ALR', 'SDR', 'SMR', 'Failure Type']

# Filter to only include data_columns
df = df[data_columns]

# Rename columns
df.columns = data_col_labems

# Do pairplot with df
sns.pairplot(df, hue='Failure Type', diag_kind='hist', palette='pastel', markers=['o', 's', 'D'], height=2.5)
plt.gcf().set_size_inches(10, 10)
plt.tight_layout()
plt.savefig('predictors.pdf')
plt.show()
