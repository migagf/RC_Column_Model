# Create files to train Surrogate GP Model

import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
# Use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load calibration_info.csv file
# The calibration_info file contains the best fit for the calibration and the resulting residual statistics
calibration_info = pd.read_csv('calibration_info.csv')

parameter_names = ['gamma', 'kappa', 'eta1', 'sig', 'lam', 'mup', 'sigp', 'rsmax', 'alpha', 'alpha1', 'alpha2', 'betam1', 'n', 'kappa_k']

# Load the data_all.csv file
data_all = pd.read_csv('data_all.csv')
# The data_all file contains the UniqueId, the FailureType, and the non-dimensional parameters representing each experiment

# Merge data_all and calibration_info using UniqueId as key
data_all = pd.merge(data_all, calibration_info, on='UniqueId')

# Sort data_all per res_median
data_all = data_all.sort_values(by='res_median')

# Plot the res_median +/- res_std for each calibration with scatter plot
res_median = data_all['res_median']
res_median_plus = res_median + data_all['res_std']
res_median_minus = res_median - data_all['res_std']

x_values = np.arange(1, len(data_all) + 1)

# Define colors based on FailureType
colors = {'Flexure': 'blue', 'Flexure-Shear': 'green', 'Shear': 'red'}

# Map FailureType to colors
data_all['color'] = data_all['FailureType'].map(colors)

# Plot with colors based on FailureType
plt.scatter(x_values, res_median, label='res_median', marker='s', c=data_all['color'], s=0.7)
plt.scatter(x_values, res_median_plus, label='res_median_plus', marker='s', c=data_all['color'], s=0.2)
plt.scatter(x_values, res_median_minus, label='res_median_minus', marker='s', c=data_all['color'], s=0.2)

# Add a horizontal line at 0.12
plt.axhline(y=0.12, color='k', linestyle='--', linewidth=1.0, alpha=0.5)
# Add text to the line
plt.text(40, 0.12, 'MAE threshold = 0.12', color='k', fontsize=10, ha='center', va='bottom')

# Add custom legend with colors
legend_labels = {'Flexure': 'blue', 'Flexure-Shear': 'green', 'Shear': 'red'}
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                       markerfacecolor=color, markersize=5) for label, color in legend_labels.items()]
plt.legend(handles=handles, title='Failure Type', loc='upper right')
plt.xlabel('Calibration Number')
plt.ylabel('Mean Average Error')
plt.grid()

plt.savefig('residuals.pdf')
plt.show()

print(data_all.columns)
# Select columns to plot
x_parameters = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']

# Create subplots for each par_y in parameter_names
'''for par_y in parameter_names:
    fig, axs = plt.subplots(1, len(x_parameters), figsize=(20, 3.5))
    
    # Plot each set of parameters
    for i, par_x in enumerate(x_parameters):
        axs[i].scatter(data_all[par_x], data_all[par_y], c=data_all['color'], s=4.0, alpha=0.3)
        axs[i].set_xlabel(par_x)
        axs[i].set_ylabel(par_y)
        axs[i].set_title(f'{par_x} vs {par_y}')
    
    plt.tight_layout()
    plt.show()'''

# Drop rows with res_median > 0.1
data_all = data_all[data_all['res_median'] <= 0.12]

# Split the data using FailureType
data_shear = pd.concat([data_all[data_all['FailureType'] == 'Shear'], data_all[data_all['FailureType'] == 'Flexure-Shear']])
data_flexure = data_all[data_all['FailureType'] == 'Flexure']

# Set predictor labels
predictors = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']

# Set output labels
outputs = ['gamma', 'kappa', 'eta1', 'sig', 'lam', 'mup', 'sigp', 'rsmax', 'alpha', 'alpha1', 'alpha2', 'betam1', 'n', 'kappa_k', 'res_min']


# Model 1: Shear
# Create a folder to store the data
os.makedirs('gpModelShear', exist_ok=True)

# Do split of the data into training and testing
from sklearn.model_selection import train_test_split
seed = 0

# Split the data into training and testing
def split_and_save(data, folder_name, seed):
    os.makedirs(folder_name, exist_ok=True)
    train = data.sample(frac=0.8, random_state=seed)
    test = data.drop(train.index)
    train.to_csv(f'{folder_name}/train.csv', index=False)
    test.to_csv(f'{folder_name}/test.csv', index=False)

    # Save only the predictors as txt file. Include the following line in the file: % ar lrr srr alr sdr smr
    with open(f'{folder_name}/input.txt', 'w') as f:
        f.write('% ' + ' '.join(predictors) + '\n')
    
    # Now include the training data into the predictors.txt file
    train[predictors].to_csv(f'{folder_name}/input.txt', sep=' ', mode='a', header=False, index=False, float_format='%.5f')

    # Save only the outputs as txt file. Include the following line in the file: % gamma kappa eta1 sig lam mup sigp rsmax alpha alpha1 alpha2 betam1 n kappa_k res_min
    with open(f'{folder_name}/output.txt', 'w') as f:
        f.write('% ' + ' '.join(outputs) + '\n')

    # Now include the training data into the outputs.txt file
    train[outputs].to_csv(f'{folder_name}/output.txt', sep=' ', mode='a', header=False, index=False, float_format='%.5f')



'''split_and_save(data_shear, 'gpModelShear', seed)
split_and_save(data_flexure, 'gpModelFlexure', seed)
split_and_save(data_all, 'gpModelAll', seed)'''


