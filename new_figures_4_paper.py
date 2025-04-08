# New figures 4 paper

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import numpy as np

# Use latex for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

figures_dir = 'Figures'
# Set seaborn style to use serif font
sns.set_theme(style="whitegrid", rc={"font.family": "serif", "text.usetex": True})


## --------
## Plot # 1
## --------
# Load the nondimentional parameter data
df = pd.read_csv('data_all.csv')

data_columns = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'FailureType']
data_col_labels = ['AR', 'LRR', 'SRR', 'ALR', 'SDR', 'SMR', 'Failure Type']

# Filter to only include data_columns
df = df[data_columns]

# Rename columns
df.columns = data_col_labels

'''# Do pairplot with df
pairplot = sns.pairplot(df, hue='Failure Type', 
                        diag_kind='hist', palette='colorblind', 
                        markers=['o', 's', 'D'], height=1.0, plot_kws={'s': 10})

plt.savefig(os.path.join(figures_dir, 'pairplot.pdf'), bbox_inches='tight')
plt.show()'''


## --------
## Plot # 2
## --------
# Reload the data_all.csv
nd_params_df = pd.read_csv('data_all.csv')

# Take columns of interest
nd_params_df = nd_params_df[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']]

# Standardize the data
scaler = StandardScaler()
nd_params_scaled = scaler.fit_transform(nd_params_df)
# Perform PCA
pca = PCA(n_components=2)
nd_params_pca = pca.fit_transform(nd_params_scaled)
# Create a DataFrame with PCA results and add the FailureType
pca_df = pd.DataFrame(data=nd_params_pca, columns=['PC1', 'PC2'])
pca_df['FailureType'] = df['Failure Type']
'''# Plot PCA results
sns.set(style="whitegrid", rc={"font.family": "serif", "text.usetex": True})
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='FailureType', palette='colorblind', s=50)
plt.title('PCA of Nondimensional Parameters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Failure Type')
plt.tight_layout()
# set axis from -4 to 4
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.savefig(os.path.join(figures_dir, 'pca_plot.pdf'), bbox_inches='tight')
plt.show()'''



## --------
## Plot # 3
## --------
# Load the calibration_info.csv
calibration_info_df = pd.read_csv('calibration_info.csv')

# Take columns of interest
calibration_info_df = calibration_info_df[['sig', 'alpha1', 'betam1', 'UniqueId']]

# Add the UniqueId to the original nondimensional parameters DataFrame
nd_params_df['UniqueId'] = calibration_info_df['UniqueId']

# Add the FailureType to the calibration_info DataFrame
calibration_info_df['FailureType'] = df['Failure Type']

'''# Create a figure with two sets of plots side by side
plt.figure(figsize=(16, 10))

# Create a grid for the main plots and the histograms
gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

# Merge nd_params_df and calibration_info_df on UniqueId
merged_df = pd.merge(nd_params_df, calibration_info_df, on='UniqueId')

# Combine 'Shear-Flexure' and 'Shear' into a single 'Shear' failure type
merged_df['FailureType'] = merged_df['FailureType'].replace({'Flexure-Shear': 'Shear'})

# Define marker shapes for each FailureType dynamically
unique_failure_types = merged_df['FailureType'].unique()
marker_shapes = {failure_type: marker for failure_type, marker in zip(unique_failure_types, ['o', 's', 'D', '^', 'v', '<', '>'])}

# Left scatter plot (AR vs Betam1)
ax_main_left = plt.subplot(gs[0, 0])
for failure_type, marker in marker_shapes.items():
    subset = merged_df[merged_df['FailureType'] == failure_type]
    ax_main_left.scatter(subset['smr'], subset['betam1'], label=failure_type, marker=marker, s=50)

ax_main_left.set_title('AR vs Betam1 Colored by FailureType')
ax_main_left.set_xlabel('Aspect Ratio (AR)')
ax_main_left.set_ylabel('Betam1')
ax_main_left.legend(title='Failure Type')
ax_main_left.grid(True)

# Left histogram (AR)
ax_hist_left = plt.subplot(gs[1, 0], sharex=ax_main_left)
for failure_type in unique_failure_types:
    subset = merged_df[merged_df['FailureType'] == failure_type]
    ax_hist_left.hist(subset['smr'], bins=20, alpha=0.6, label=failure_type, stacked=True)

ax_hist_left.set_xlabel('Aspect Ratio (AR)')
ax_hist_left.set_ylabel('Frequency')
ax_hist_left.legend(title='Failure Type')
ax_hist_left.grid(True)

# Right scatter plot (SMR vs Sig)
ax_main_right = plt.subplot(gs[0, 1])
for failure_type, marker in marker_shapes.items():
    subset = merged_df[merged_df['FailureType'] == failure_type]
    ax_main_right.scatter(subset['smr'], subset['sig'], label=failure_type, marker=marker, s=50)

ax_main_right.set_title('SMR vs Sig Colored by FailureType')
ax_main_right.set_xlabel('SMR')
ax_main_right.set_ylabel('Sig')
ax_main_right.legend(title='Failure Type')
ax_main_right.grid(True)

# Right histogram (SMR)
ax_hist_right = plt.subplot(gs[1, 1], sharex=ax_main_right)
for failure_type in unique_failure_types:
    subset = merged_df[merged_df['FailureType'] == failure_type]
    ax_hist_right.hist(subset['smr'], bins=20, alpha=0.6, label=failure_type, stacked=True)

ax_hist_right.set_xlabel('SMR')
ax_hist_right.set_ylabel('Frequency')
ax_hist_right.legend(title='Failure Type')
ax_hist_right.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'scatter_histogram_plot.pdf'), bbox_inches='tight')
plt.show()'''




## --------
## Plot # 4
## --------
# Do PCA on the calibration_info_df

# Reload the calibration_info_df
calibration_info_df = pd.read_csv('calibration_info.csv')
# Take columns of interest
calibration_info_df = calibration_info_df[['gamma', 'kappa', 'eta1', 'sig', 'lam', 'mup', 'sigp', 'rsmax', 'alpha', 'alpha1', 'alpha2', 'betam1', 'n', 'kappa_k', 'UniqueId']]

# Perform PCA on calibration_info_df
scaler = StandardScaler()
calibration_info_scaled = scaler.fit_transform(calibration_info_df.drop(columns=['UniqueId']))
pca_calibration = PCA(n_components=1)
calibration_info_pca = pca_calibration.fit_transform(calibration_info_scaled)
calibration_info_df['PCA_Calibration'] = calibration_info_pca

# Perform PCA on nd_params_df
nd_params_df = pd.read_csv('data_all.csv')
nd_params_df = nd_params_df[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'UniqueId', 'FailureType']]
scaler = StandardScaler()
nd_params_scaled = scaler.fit_transform(nd_params_df.drop(columns=['UniqueId', 'FailureType']))
pca_nd_params = PCA(n_components=1)
nd_params_pca = pca_nd_params.fit_transform(nd_params_scaled)
nd_params_df['PCA_ND_Params'] = nd_params_pca

# Merge the two DataFrames on UniqueId
merged_pca_df = pd.merge(calibration_info_df[['UniqueId', 'PCA_Calibration']], 
                         nd_params_df[['UniqueId', 'PCA_ND_Params', 'FailureType']], 
                         on='UniqueId')

# Create a figure with scatter plot and histogram
plt.figure(figsize=(6, 6))
gs = GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.3)

# Scatter plot (PCA_ND_Params vs PCA_Calibration)
ax_scatter = plt.subplot(gs[0, 0])
sns.scatterplot(data=merged_pca_df, x='PCA_ND_Params', y='PCA_Calibration', 
                hue='FailureType', palette='colorblind', s=50, ax=ax_scatter)
ax_scatter.set_xlabel('ND-Parameters PCA (1st component)')
ax_scatter.set_ylabel('BW-Parameters PCA (1st component)')
ax_scatter.grid(True)

# Move legend outside the plot
ax_scatter.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax_scatter.set_xlim(-4, 4)
ax_scatter.set_ylim(-4, 4)

# Histogram (PCA_ND_Params)
ax_hist = plt.subplot(gs[1, 0], sharex=ax_scatter)
for failure_type in merged_pca_df['FailureType'].unique():
    subset = merged_pca_df[merged_pca_df['FailureType'] == failure_type]
    ax_hist.hist(subset['PCA_ND_Params'], bins=20, alpha=0.6, label=failure_type, stacked=True)

ax_hist.set_xlabel('ND-Parameters PCA (1st component)')
ax_hist.set_ylabel('Frequency')
ax_hist.grid(True)

# Move legend outside the histogram
ax_hist.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.subplots_adjust(right=0.7)  # Adjust the right margin to make space for the legend

# plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'pca_scatter_histogram_plot.pdf'), bbox_inches='tight')
plt.show()



'''
## --------
## Plot # 5
## --------
# Load the error metrics data for model 0, 1, 2, 3, 4, 5, avg
error_metrics_0 = pd.read_csv('error_metrics_model_0.csv')
error_metrics_1 = pd.read_csv('error_metrics_model_1.csv')
error_metrics_2 = pd.read_csv('error_metrics_model_2.csv')
error_metrics_3 = pd.read_csv('error_metrics_model_3.csv')
error_metrics_4 = pd.read_csv('error_metrics_model_4.csv')
error_metrics_5 = pd.read_csv('error_metrics_model_5.csv')
error_metrics_a = pd.read_csv('error_metrics_model_avg.csv')

# Remove points with error > 0.5
error_metrics_0 = error_metrics_0[error_metrics_0['err_surr_exp'] < 0.5]
error_metrics_1 = error_metrics_1[error_metrics_1['err_surr_exp'] < 0.5]
error_metrics_2 = error_metrics_2[error_metrics_2['err_surr_exp'] < 0.5]
error_metrics_3 = error_metrics_3[error_metrics_3['err_surr_exp'] < 0.5]
error_metrics_4 = error_metrics_4[error_metrics_4['err_surr_exp'] < 0.5]
error_metrics_5 = error_metrics_5[error_metrics_5['err_surr_exp'] < 0.5]
error_metrics_a = error_metrics_a[error_metrics_a['err_surr_exp'] < 0.5]

sel_err_metric = 'err_surr_exp'

# Create a figure with two rows of subplots (PDFs on top, CDFs below)
fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1, 'wspace': 0.1})

# Define a base color and generate shades for the first 6 models
base_color = 'blue'
shades = sns.light_palette(base_color, n_colors=6, reverse=True)

# Function to fit and plot KDE for each model
def plot_kde(data, label, color, ax_pdf, ax_cdf):
    kde = gaussian_kde(data)
    x_vals = np.linspace(0.0, 0.3, 1000)  # Set x range from 0.0 to 0.3
    y_vals_pdf = kde(x_vals)
    y_vals_cdf = np.cumsum(y_vals_pdf) / np.sum(y_vals_pdf)  # Compute CDF from PDF

    # Plot PDF
    ax_pdf.fill_between(x_vals, y_vals_pdf, alpha=0.4, label=label, color=color)
    ax_pdf.plot(x_vals, y_vals_pdf, color=color, linewidth=2)

    # Plot CDF
    ax_cdf.plot(x_vals, y_vals_cdf, color=color, linewidth=2, label=label)

# Plot KDEs and CDFs for the first 6 models
for i, (data, label, color) in enumerate(zip(
        [error_metrics_0, error_metrics_1, error_metrics_2, error_metrics_3, error_metrics_4, error_metrics_5],
        ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
        shades)):
    plot_kde(data[sel_err_metric], label, color, axes[0, 0], axes[1, 0])

# Add labels, title, and legend to the first column (Models 0-5)
axes[0, 0].set_ylabel('KDE Density', fontsize=14)
axes[0, 0].legend(fontsize=12, title_fontsize=14)
axes[0, 0].grid(True, linestyle='--', alpha=0.7)
axes[0, 0].set_xlim(0.0, 0.3)  # Set x-axis limits
# Remove x tick labels
axes[0, 0].set_xticklabels([])

axes[1, 0].set_xlabel('Mean Absolute Error', fontsize=14)
axes[1, 0].set_ylabel('CDF', fontsize=14)
axes[1, 0].legend(fontsize=12, title_fontsize=14)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].set_xlim(0.0, 0.3)  # Set x-axis limits

# Plot KDE and CDF for the average model in the second column
plot_kde(error_metrics_a[sel_err_metric], 'Final GP', 'gray', axes[0, 1], axes[1, 1])

# Add labels, title, and legend to the second column (Model Avg)
axes[0, 1].legend(fontsize=12, title_fontsize=14)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].set_xlim(0.0, 0.3)  # Set x-axis limits
# remove y tick labels and x tick labels
axes[0, 1].set_yticklabels([])
axes[0, 1].set_xticklabels([])

axes[1, 1].set_xlabel('Mean Absolute Error', fontsize=14)
axes[1, 1].legend(fontsize=12, title_fontsize=14)
axes[1, 1].grid(True, linestyle='--', alpha=0.7)
axes[1, 1].set_xlim(0.0, 0.3)  # Set x-axis limits
# remove y tick labels
axes[1, 1].set_yticklabels([])

# Adjust layout for better spacing
plt.tight_layout()

# Uncomment to save the plot
# plt.savefig('error_metrics_pdf_cdf.pdf')
# Show the plot
plt.savefig(os.path.join(figures_dir, 'error_metrics_pdf_cdf.pdf'), bbox_inches='tight')
plt.show()
'''

'''
## --------
## Plot # 6
## --------
# Repeat the same process for other error metrics
for sel_err_metric, filename in zip(['err_surr_cal'], 
                                    ['error_metrics_pdf_cdf_surr_cal.pdf']):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1, 'wspace': 0.1})
    for i, (data, label, color) in enumerate(zip(
            [error_metrics_0, error_metrics_1, error_metrics_2, error_metrics_3, error_metrics_4, error_metrics_5],
            ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
            shades)):
        plot_kde(data[sel_err_metric], label, color, axes[0, 0], axes[1, 0])
    axes[0, 0].set_ylabel('KDE Density', fontsize=14)
    axes[0, 0].legend(fontsize=12, title_fontsize=14)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].set_xlim(0.0, 0.3)  # Set x-axis limits
    axes[0, 0].set_xticklabels([])  # Remove x tick labels

    axes[1, 0].set_xlabel('Mean Absolute Error', fontsize=14)
    axes[1, 0].set_ylabel('CDF', fontsize=14)
    axes[1, 0].legend(fontsize=12, title_fontsize=14)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].set_xlim(0.0, 0.3)  # Set x-axis limits

    plot_kde(error_metrics_a[sel_err_metric], 'Final GP', 'gray', axes[0, 1], axes[1, 1])
    axes[0, 1].legend(fontsize=12, title_fontsize=14)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].set_xlim(0.0, 0.3)  # Set x-axis limits
    axes[0, 1].set_yticklabels([])  # Remove y tick labels
    axes[0, 1].set_xticklabels([])  # Remove x tick labels

    axes[1, 1].set_xlabel('Mean Absolute Error', fontsize=14)
    axes[1, 1].legend(fontsize=12, title_fontsize=14)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].set_xlim(0.0, 0.3)  # Set x-axis limits
    axes[1, 1].set_yticklabels([])  # Remove y tick labels

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), bbox_inches='tight')
    plt.show()



## --------
## Plot # 7
## --------
# Repeat the last two plots, but now split the data by FailureType
# No need to reload the data.

# Lets start for FailureType = Flexure
# Filter the data for Flexure

err_metrics_0_flex = error_metrics_0[error_metrics_0['FailureMode'] == 'Flexure']
err_metrics_1_flex = error_metrics_1[error_metrics_1['FailureMode'] == 'Flexure']
err_metrics_2_flex = error_metrics_2[error_metrics_2['FailureMode'] == 'Flexure']
err_metrics_3_flex = error_metrics_3[error_metrics_3['FailureMode'] == 'Flexure']
err_metrics_4_flex = error_metrics_4[error_metrics_4['FailureMode'] == 'Flexure']
err_metrics_5_flex = error_metrics_5[error_metrics_5['FailureMode'] == 'Flexure']
err_metrics_a_flex = error_metrics_a[error_metrics_a['FailureMode'] == 'Flexure']
# Create a figure with two rows of subplots (PDFs on top, CDFs below)
fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1, 'wspace': 0.1})

# Define a base color and generate shades for the first 6 models
base_color = 'green'
shades = sns.light_palette(base_color, n_colors=6, reverse=True)

# Plot KDEs and CDFs for the first 6 models
for i, (data, label, color) in enumerate(zip(
    [err_metrics_0_flex, err_metrics_1_flex, err_metrics_2_flex, err_metrics_3_flex, err_metrics_4_flex, err_metrics_5_flex],
    ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
    shades)):
    plot_kde(data[sel_err_metric], label, color, axes[0, 0], axes[1, 0])

# Add labels, title, and legend to the first column (Models 0-5)
axes[0, 0].set_ylabel('KDE Density', fontsize=14)
axes[0, 0].legend(fontsize=12, title_fontsize=14)
axes[0, 0].grid(True, linestyle='--', alpha=0.7)
axes[0, 0].set_xlim(0.0, 0.3)  # Set x-axis limits
axes[0, 0].set_xticklabels([])  # Remove x tick labels

axes[1, 0].set_xlabel('Mean Absolute Error', fontsize=14)
axes[1, 0].set_ylabel('CDF', fontsize=14)
axes[1, 0].legend(fontsize=12, title_fontsize=14)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].set_xlim(0.0, 0.3)  # Set x-axis limits

# Plot KDE and CDF for the average model in the second column
plot_kde(err_metrics_a_flex[sel_err_metric], 'Final GP', 'gray', axes[0, 1], axes[1, 1])

# Add labels, title, and legend to the second column (Model Avg)
axes[0, 1].legend(fontsize=12, title_fontsize=14)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].set_xlim(0.0, 0.3)  # Set x-axis limits
axes[0, 1].set_yticklabels([])  # Remove y tick labels
axes[0, 1].set_xticklabels([])  # Remove x tick labels

axes[1, 1].set_xlabel('Mean Absolute Error', fontsize=14)
axes[1, 1].legend(fontsize=12, title_fontsize=14)
axes[1, 1].grid(True, linestyle='--', alpha=0.7)
axes[1, 1].set_xlim(0.0, 0.3)  # Set x-axis limits
axes[1, 1].set_yticklabels([])  # Remove y tick labels

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'error_metrics_pdf_cdf_flexure.pdf'), bbox_inches='tight')
plt.show()



## --------
## Plot # 8
## --------
# Repeat the same process for FailureType = Shear or Flexure-Shear
# Filter the data for Shear
err_metrics_0_shear = error_metrics_0[error_metrics_0['FailureMode'].isin(['Shear', 'Flexure-Shear'])]
err_metrics_1_shear = error_metrics_1[error_metrics_1['FailureMode'].isin(['Shear', 'Flexure-Shear'])]
err_metrics_2_shear = error_metrics_2[error_metrics_2['FailureMode'].isin(['Shear', 'Flexure-Shear'])]
err_metrics_3_shear = error_metrics_3[error_metrics_3['FailureMode'].isin(['Shear', 'Flexure-Shear'])]
err_metrics_4_shear = error_metrics_4[error_metrics_4['FailureMode'].isin(['Shear', 'Flexure-Shear'])]
err_metrics_5_shear = error_metrics_5[error_metrics_5['FailureMode'].isin(['Shear', 'Flexure-Shear'])]
err_metrics_a_shear = error_metrics_a[error_metrics_a['FailureMode'].isin(['Shear', 'Flexure-Shear'])]

# Create a figure with two rows of subplots (PDFs on top, CDFs below)
fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1, 'wspace': 0.1})

# Define a base color and generate shades for the first 6 models
base_color = 'red'
shades = sns.light_palette(base_color, n_colors=6, reverse=True)

# Plot KDEs and CDFs for the first 6 models
for i, (data, label, color) in enumerate(zip(
    [err_metrics_0_shear, err_metrics_1_shear, err_metrics_2_shear, err_metrics_3_shear, err_metrics_4_shear, err_metrics_5_shear],
    ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
    shades)):
    plot_kde(data[sel_err_metric], label, color, axes[0, 0], axes[1, 0])

# Add labels, title, and legend to the first column (Models 0-5)
axes[0, 0].set_ylabel('KDE Density', fontsize=14)
axes[0, 0].legend(fontsize=12, title_fontsize=14)
axes[0, 0].grid(True, linestyle='--', alpha=0.7)
axes[0, 0].set_xlim(0.0, 0.3)  # Set x-axis limits
axes[0, 0].set_xticklabels([])  # Remove x tick labels

axes[1, 0].set_xlabel('Mean Absolute Error', fontsize=14)
axes[1, 0].set_ylabel('CDF', fontsize=14)
axes[1, 0].legend(fontsize=12, title_fontsize=14)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].set_xlim(0.0, 0.3)  # Set x-axis limits

# Plot KDE and CDF for the average model in the second column
plot_kde(err_metrics_a_shear[sel_err_metric], 'Final GP', 'gray', axes[0, 1], axes[1, 1])

# Add labels, title, and legend to the second column (Model Avg)
axes[0, 1].legend(fontsize=12, title_fontsize=14)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].set_xlim(0.0, 0.3)  # Set x-axis limits
axes[0, 1].set_yticklabels([])  # Remove y tick labels
axes[0, 1].set_xticklabels([])  # Remove x tick labels

axes[1, 1].set_xlabel('Mean Absolute Error', fontsize=14)
axes[1, 1].legend(fontsize=12, title_fontsize=14)
axes[1, 1].grid(True, linestyle='--', alpha=0.7)
axes[1, 1].set_xlim(0.0, 0.3)  # Set x-axis limits
axes[1, 1].set_yticklabels([])  # Remove y tick labels

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'error_metrics_pdf_cdf_shear.pdf'), bbox_inches='tight')
plt.show()'''
