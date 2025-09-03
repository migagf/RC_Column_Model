# Create files to train Surrogate GP Model

import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt

# Use latex for plot
plt.rc('text', usetex=True)  # Use LaTeX for rendering text in plots
plt.rc('font', family='serif')  # Use serif font for LaTeX text

def create_train_files(data, split, output_dir, labels, logfile):
    predictors = labels['predictors']
    outputs = labels['outputs']

    # from data, extract the predictor columns as output
    x_train = data[predictors]
    y_train = data[outputs]

    # Remove any input.txt and output.txt files
    for file in os.listdir(output_dir):
        if file.startswith('input') or file.startswith('output'):
            os.remove(os.path.join(output_dir, file))

    # Take these and save as tab separated files
    x_train.to_csv(f'{output_dir}/input.txt', sep=' ', index=False, header=False, float_format='%.5f')
    y_train.to_csv(f'{output_dir}/output.txt', sep=' ', index=False, header=False, float_format='%.5f')

    # Open the files and add a '%' symbol at the beggining of each
    with open(f'{output_dir}/input.txt', 'r') as f:
        lines = f.readlines()
    with open(f'{output_dir}/input.txt', 'w') as f:
        f.write('% ' + ' '.join(predictors) + '\n')
        f.writelines(lines)

    with open(f'{output_dir}/output.txt', 'r') as f:
        lines = f.readlines()
    with open(f'{output_dir}/output.txt', 'w') as f:
        f.write('% ' + ' '.join(outputs) + '\n')
        f.writelines(lines)

    # Write to log
    write_to_log(f'Created training files for split {split}', logfile)

    pass


# Split the data into training and testing
def k_fold_split(data, output_folder, labels, seed, number_of_splits, logfile):
    
    # Shuffle data in the rows
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    write_to_log(f'Data shuffled with seed {seed}', logfile)

    # Add a 'split' column for data
    data['split'] = pd.qcut(data.index, q=number_of_splits, labels=False)
    write_to_log(f'Data shuffled and split into {number_of_splits} parts', logfile)

    # Write to log how many samples are in each split
    for i in range(number_of_splits):
        n_samples = data[data['split'] == i].shape[0]
        write_to_log(f'  Split {i}: {n_samples} samples', logfile)

    # Save the data with the split into the output folder
    os.makedirs(output_folder, exist_ok=True)
    data.to_csv(f'{output_folder}/data_split.csv', index=False)

    # Create training input files for quoFEM
    for split in range(number_of_splits):
        write_to_log(f'Creating training files for split {split}', logfile)

        # Create a folder named split
        split_out_dir = f'{output_folder}/split_{str(split).zfill(2)}'
        os.makedirs(split_out_dir, exist_ok=True)

        # Training data are all samples not in the current split
        create_train_files(
            data[data['split'] != split].drop(columns=['split']), split, split_out_dir, labels, logfile)
    
    # Create no split training data (all data used to train the model)
    write_to_log(f'Creating training files for no split (all data)', logfile)
    no_split_out_dir = f'{output_folder}/no_split'
    os.makedirs(no_split_out_dir, exist_ok=True)

    # Training data are all samples
    create_train_files(
        data.drop(columns=['split']), 'no_split', no_split_out_dir, labels, logfile)

    '''
    os.makedirs(folder_name, exist_ok=True)
    train = data.sample(frac=0.75, random_state=seed)
    test = data.drop(train.index)

    train.to_csv(f'{folder_name}/train{str(shuffle_id)}.csv', index=False)
    test.to_csv(f'{folder_name}/test{str(shuffle_id)}.csv', index=False)

    # Adding column names: % ar lrr srr alr sdr smr
    with open(f'{folder_name}/input{str(shuffle_id)}.txt', 'w') as f:
        f.write('% ' + ' '.join(predictors) + '\n')

    # Adding training data to the input.txt file
    train[predictors].to_csv(f'{folder_name}/input{str(shuffle_id)}.txt', sep=' ', mode='a', header=False, index=False, float_format='%.5f')

    # Adding column names: % gamma kappa eta1 sig lam mup sigp rsmax alpha alpha1 alpha2 betam1 n kappa_k res_min
    with open(f'{folder_name}/output{str(shuffle_id)}.txt', 'w') as f:
        f.write('% ' + ' '.join(outputs) + '\n')
    # Now include the training data into the outputs.txt file
    train[outputs].to_csv(f'{folder_name}/output{str(shuffle_id)}.txt', sep=' ', mode='a', header=False, index=False, float_format='%.5f')
'''

def write_to_log(message, logfile):
    with open(logfile, 'a') as f:
        f.write(message + '\n')
    print(message)


if __name__ == "__main__":
    # Main settings
    showplots = False
    logfile = 'gp_training_data/processed/log.txt'
    nsurr = 6   # Number of surrogate models to create for cross-validation
    median_threshold = 0.1
    seed = 99  # random seed

    # Initializing log-file
    os.remove(logfile)
    write_to_log(f'Creating {nsurr} surrogate models for cross-validation', logfile)

    # :::
    # Load Files
    # :::
    
    # The calibration_info file contains the best fit for the calibration and the resulting residual statistics
    calibration_info = pd.read_csv('gp_training_data/calibrations/calibration_info.csv')
    write_to_log(f'Successfully loaded calibration_info with: {len(calibration_info)} calibration records', logfile)

    # Load the DataAll_NDonly.csv file. This file contains the non-dimensional parameters for all experiments (rectangular and spiral)
    data_all = pd.read_csv('gp_training_data/raw/DataAll_NDonly.csv')
    write_to_log(f'Successfully loaded DataAll_NDonly with: {len(data_all)} records', logfile)

    # Merge data_all and calibration_info using UniqueId as key and sort by res_median
    data_all = pd.merge(data_all, calibration_info, on='UniqueId')
    
    # Remove unnamed columns
    data_all = data_all.loc[:, ~data_all.columns.str.contains('^Unnamed')]
    # Rename Name_x column as Name and remove Name_y column
    data_all = data_all.rename(columns={'Name_x': 'Name'})
    data_all = data_all.loc[:, ~data_all.columns.str.contains('^Name_y')]
    data_all = data_all.sort_values(by='res_median')

    # ::: Plot #1 - Residual median and std for calibrations :::
    if showplots:
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
        plt.figure(figsize=(6, 4))
        plt.scatter(x_values, res_median, label='res_median', marker='s', c=data_all['color'], s=0.7)
        plt.scatter(x_values, res_median_plus, label='res_median_plus', marker='s', c=data_all['color'], s=0.2)
        plt.scatter(x_values, res_median_minus, label='res_median_minus', marker='s', c=data_all['color'], s=0.2)

        # Add a horizontal line at 0.10 with a text box
        plt.text(10, 0.11, 'Threshold at MAE=0.1',
                horizontalalignment='left', verticalalignment='center', fontsize=10, color='k',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        plt.axhline(y=0.1, color='k', linestyle='--', linewidth=1.0)

        # Add the normalized cumulative number of tests as a grey area plot
        cumulative_tests = np.arange(1, len(data_all) + 1)
        cumulative_tests_norm = cumulative_tests / cumulative_tests[-1] * 0.25  # Normalize to [0, 0.25]
        plt.fill_between(x_values, 0, cumulative_tests_norm, color='grey', alpha=0.1, label='Cumulative Tests')

        # Add a second y-axis for the normalized cumulative number of tests
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of tests', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.set_ylim([0, 0.25])
        ax2.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
        ax2.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])

        # Create a custom legend for FailureType
        legend_labels = {'Flexure': 'Flexure', 'Flexure-Shear': 'Flexure-Shear', 'Shear': 'Shear'}
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5, label=label) 
                for label, color in colors.items()]
        ax1.set_xlabel('Test \# (Sorted by MAE)')  # X-axis label
        ax1.set_ylabel('Mean Absolute Error')  # Y-axis label
        ax1.set_xlim([0, 300])
        ax1.set_ylim([0, 0.25])
        ax1.grid()
        ax1.legend(handles=legend_handles, title="Failure Type")
        plt.savefig('Figures/residuals_plot.pdf')  # Save the figure to a pdf file
        plt.show()

    # ::: Plot #2 - Parameter correlations :::
    if showplots:
        # Select columns to plot
        x_parameters = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']
        x_parameter_labels = ['AR', 'LRR', 'SRR', 'ALR', 'SDR', 'SR']

        # Names of the Bouc-Wen model parameters
        parameter_names = [
            'gamma', 'kappa', 'eta1', 'sig', 'lam', 'mup', 'sigp', 
            'rsmax', 'alpha', 'alpha1', 'alpha2', 'betam1', 'n', 'kappa_k']
        
        # Create subplots for each par_y in parameter_names
        for par_y in parameter_names:
            fig, axs = plt.subplots(1, len(x_parameters), figsize=(20, 3.5))
            
            # Plot each set of parameters
            for i, par_x in enumerate(x_parameters):
                axs[i].scatter(data_all[par_x], data_all[par_y], c=data_all['color'], s=4.0, alpha=0.3)
                axs[i].set_xlabel(par_x)
                axs[i].set_ylabel(par_y)
                axs[i].set_title(f'{par_x} vs {par_y}')
            
            plt.tight_layout()
            plt.show()

    # :::
    # Data Filtering
    # :::

    # Drop rows with res_median > median_threshold
    data_all = data_all[data_all['res_median'] <= median_threshold]

    # Print how many rows are left after filtering by res_median <= median_threshold
    write_to_log(f'Number of rows after filtering by res_median <= {median_threshold}: {len(data_all)}', logfile)

    # Split the data using FailureType
    data_shear = pd.concat([data_all[data_all['FailureType'] == 'Shear'], data_all[data_all['FailureType'] == 'Flexure-Shear']])
    data_flexure = data_all[data_all['FailureType'] == 'Flexure']

    # Print how many rows are left in data_shear and data_flexure:
    write_to_log(f'Number of shear influenced tests: {len(data_shear)}', logfile)
    write_to_log(f'Number of flexure tests: {len(data_flexure)}', logfile)

    # :::
    # Split and create training sets
    # :::

    # Set predictor labels
    labels = {
        'predictors': ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr'],
        'outputs': ['gamma', 'kappa', 'eta1', 'sig', 'lam', 'mup', 'sigp', 'rsmax', 'alpha', 'alpha1', 'alpha2', 'betam1', 'n', 'kappa_k', 'res_min']
    }

    # Shear model
    k_fold_split(data_shear, 'gp_training_data/processed/gpModelShear', labels=labels, seed=seed, number_of_splits=nsurr, logfile=logfile)

    # Flexure model
    k_fold_split(data_flexure, 'gp_training_data/processed/gpModelFlexure', labels=labels, seed=seed, number_of_splits=nsurr, logfile=logfile)

    # Get correlation matrix for the predictors from all tests
    corr_matrix = data_all[labels['predictors']].corr()

    # Save the correlation matrix as a csv without the column titles and no index
    corr_matrix.to_csv('gp_training_data/processed/correlation_matrix.csv', header=False, index=False)

    # Plot the correlation matrix using heatmap
    if True:
        plt.figure(figsize=(4, 4))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix')
        plt.show()

    # Once this is done, use quoFEM to run the surrogate model training.
    # ::: (Nothing else here) :::