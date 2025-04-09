# Code to create figure of TMCMC prior and posterior distributions

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

figures_dir = 'Figures'

# Use latex for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# be0d0f41-2ce9-457a-b04e-331c99c615b4-007
folder = r'D:\tacc scratch\25_03\13\be0d0f41-2ce9-457a-b04e-331c99c615b4-007'

results_folder = folder + r'\results'
# load resultsStage0.csv without a header
results_stage0 = pd.read_csv(results_folder + r'\resultsStage0.csv', header=None)

# load resultsStage25.csv without a header
results_stage25 = pd.read_csv(results_folder + r'\resultsStage25.csv', header=None)

# load cal_file.csv
cal_file = pd.read_csv(folder + r'\cal_file.csv', header=None)

# read the first row of dakotaTab.out file, and use these values (separated by spaces) as the column names
dakota_tab_out = results_folder + r'\dakotaTab.out'
with open(dakota_tab_out, 'r') as f:
    column_names = f.readline().strip().split()

# Ignore the first two elements in column_names
column_names = column_names[2:]

# Apply the modified column names to results_stage0 and results_stage25
results_stage0.columns = column_names
results_stage25.columns = column_names

# Print column names
print(column_names[0:15])

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Plot a histogram with the values in the sig column for the resultsStage0.csv file and on top of it plot a histogram with the values in the sig column for the resultsStage25.csv file
axes[0].hist(results_stage0['sig'], bins=30, alpha=0.5, label='Samples from prior', color='blue', density=True)
axes[0].hist(results_stage25['sig'], bins=15, alpha=0.5, label='Samples from posterior', color='red', density=True)
axes[0].set_xlabel(r'Bouc-Wen Parameter $\sigma$')
axes[0].set_ylabel('Density')
axes[0].legend()

# Plot a histogram with the values in the betam1 column for the resultsStage0.csv file and on top of it plot a histogram with the values in the betam1 column for the resultsStage25.csv file
axes[1].hist(results_stage0['betam1'], bins=30, alpha=0.5, label='Samples from prior', color='blue', density=True)
axes[1].hist(results_stage25['betam1'], bins=15, alpha=0.5, label='Samples from posterior', color='red', density=True)
axes[1].set_xlabel(r'Bouc-Wen Parameter $\beta_{m,1}$')
axes[1].set_ylabel('Density')
axes[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
# Save the figure to the figures directory
plt.savefig(f'{figures_dir}/prior_posterior_distributions.pdf')
plt.show()


# Now, let's create a new plot with the distributions of the force vectors. Need to sample.
# Force vectors start at column 16

# Get number of rows in results_stage0
num_rows_stage0 = results_stage0.shape[0]

# Sample 10% of the rows from results_stage0 and results_stage25
sample_size = int(num_rows_stage0 * 0.1)
sample_stage0 = results_stage0.sample(n=sample_size, random_state=42)
sample_stage25 = results_stage25.sample(n=sample_size, random_state=42)

# Create a figure with two subplots side by side for force vectors
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Plot lines for each force vector in the sample_stage0 and sample_stage25
for i in range(sample_size):
    axes[0].plot(sample_stage0.iloc[i, 16:], '.-', alpha=0.2, color='blue')
    axes[1].plot(sample_stage25.iloc[i, 16:], '.-', alpha=0.2, color='red')

axes[0].set_title('Force Vectors from Prior Samples')
axes[1].set_title('Force Vectors from Posterior Samples')

# Add values  from cal_file to each subplot
axes[0].plot( cal_file.iloc[0, :], 'k-', linewidth=1.0)
axes[1].plot( cal_file.iloc[0, :], 'k-', linewidth=1.0)

# Delete xticks for the first subplot
axes[0].set_xticks([])
# Define x_ticks for the second subplot with values from 0 to the length of the vector with a step of 50
x_ticks = np.arange(0, sample_stage25.shape[1] - 16, 200)
axes[1].set_xticks(x_ticks)
# Set x_tick labels to the corresponding values
axes[1].set_xticklabels(x_ticks)
# Set xlabel
axes[1].set_xlabel('Pseudo-time step')
# Define y labels as Force (kN) for both subplots
axes[0].set_ylabel('Force (kN)')
# Set ylim to [-200, 200]
axes[0].set_ylim(-200, 200)
axes[1].set_ylabel('Force (kN)')
axes[1].set_ylim(-200, 200)
# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(f'{figures_dir}/force_vectors_prior_posterior.pdf')
plt.show()