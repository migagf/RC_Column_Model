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
data_col_labels = ['AR', 'LRR', 'SRR', 'ALR', 'SDR', 'SMR', 'Failure Type']

# Filter to only include data_columns
df = df[data_columns]

# Rename columns
df.columns = data_col_labels

# Do pairplot with df
pairplot = sns.pairplot(df, hue='Failure Type', 
                        diag_kind='hist', palette='colorblind', 
                        markers=['o', 's', 'D'], height=1.0, plot_kws={'s': 10})

#plt.savefig('predictors.pdf')
#plt.show()



# Reload the data_all.csv

nd_params_df = pd.read_csv('data_all.csv')

# Load the calibration_info.csv
calibration_info_df = pd.read_csv('calibration_info.csv')

# Merge the two dataframes on the 'UniqueId' column
merged_df = pd.merge(nd_params_df, calibration_info_df, on='UniqueId')

sel_columns = ['ar', 'smr', 'sig', 'alpha1', 'betam1', 'FailureType']

# Filter to only include sel_columns
merged_df = merged_df[sel_columns]

# Rename columns
merged_df.columns = ['ar', 'srr', 'smr', 'sigma', 'alpha1', 'betam1', 'Failure Type']

# Do plots:
# ar vs sigma, ar vs alpha1, ar vs betam1
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.scatterplot(data=merged_df, x='ar', y='sigma', hue='Failure Type', ax=axes[0])
axes[0].set_title('AR vs Sigma')
sns.scatterplot(data=merged_df, x='ar', y='alpha1', hue='Failure Type', ax=axes[1])
axes[1].set_title('AR vs Alpha1')
sns.scatterplot(data=merged_df, x='ar', y='betam1', hue='Failure Type', ax=axes[2])
axes[2].set_title('AR vs Betam1')
plt.tight_layout()
plt.show()
