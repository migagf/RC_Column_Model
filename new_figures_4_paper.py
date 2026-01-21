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
import shutil


# Use latex for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['hatch.linewidth'] = 0.6  # make hatch patterns visible in print

# :::
# Functions
# :::
# Function to fit and plot KDE for each model
def plot_kde(data, label, color, ax_pdf, ax_cdf):
    kde = gaussian_kde(data)
    x_vals = np.linspace(0.0, 0.5, 1000)  # Set x range from 0.0 to 0.3
    y_vals_pdf = kde(x_vals)
    y_vals_cdf = np.cumsum(y_vals_pdf) / np.sum(y_vals_pdf)  # Compute CDF from PDF

    # Plot PDF
    ax_pdf.fill_between(x_vals, y_vals_pdf, alpha=0.4, label=label, color=color)
    ax_pdf.plot(x_vals, y_vals_pdf, color=color, linewidth=2)

    # Plot CDF
    ax_cdf.plot(x_vals, y_vals_cdf, color=color, linewidth=2, label=label)


figures_dir = 'Figures'
# Set seaborn style to use serif font
sns.set_theme(style="whitegrid", rc={"font.family": "serif", "text.usetex": True})

data_path = 'gp_training_data/raw/DataAll_NDonly.csv'

do_plots = [1]

# Load the nondimentional parameter data
df = pd.read_csv(data_path)

data_columns = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'FailureType']
data_col_labels = ['AR', 'LRR', 'SRR', 'ALR', 'SDR', 'SMR', 'Failure Type']

# Filter to only include data_columns
df = df[data_columns]

# Rename columns
df.columns = data_col_labels


## --------
## Plot # 1
## --------
if 1 in do_plots:
    # Do pairplot with df using a greyscale palette
    n_types = df['Failure Type'].nunique()
    grey_palette = sns.color_palette("Greys", n_colors=n_types)
    markers = ['o', 's', 'v', '^', 'v', '<', '>'][:n_types]
    pairplot = sns.pairplot(df, hue='Failure Type',
                            diag_kind='kde', palette=grey_palette,
                            markers=markers, height=1.0, plot_kws={'s': 20, 'alpha': 0.7}, diag_kws={'shade': True})
    # remove grid lines
    for ax in pairplot.axes.flatten():
        ax.grid(False)

    plt.savefig(os.path.join(figures_dir, 'pairplot.pdf'), bbox_inches='tight')
    
    plt.show()


## --------
## Plot # 2
## --------
if 2 in do_plots:

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


if 3 in do_plots:
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


if 4 in do_plots:
    ## --------
    ## Plot # 4
    ## --------
    # Do PCA on the calibration_info_df

    # Reload the calibration_info_df
    calibration_info_df = pd.read_csv('gp_training_data/calibrations/calibration_info.csv')

    # Take columns of interest
    calibration_info_df = calibration_info_df[['gamma', 'kappa', 'eta1', 'sig', 'lam', 'mup', 'sigp', 'rsmax', 'alpha', 'alpha1', 'alpha2', 'betam1', 'n', 'kappa_k', 'UniqueId']]

    # Perform PCA on calibration_info_df
    scaler = StandardScaler()
    calibration_info_scaled = scaler.fit_transform(calibration_info_df.drop(columns=['UniqueId']))
    pca_calibration = PCA(n_components=1)
    calibration_info_pca = pca_calibration.fit_transform(calibration_info_scaled)
    calibration_info_df['PCA_Calibration'] = calibration_info_pca

    # Perform PCA on nd_params_df
    nd_params_df = pd.read_csv('gp_training_data/raw/DataAll_NDonly.csv')
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
    # Plot each failure type with a different marker
    failure_types = merged_pca_df['FailureType'].unique()
    markers = ['o', 's', 'D', '^', 'v', '<', '>'][:len(failure_types)]
    palette = sns.color_palette("Greys", n_colors=len(failure_types))
    for ft, m, c in zip(failure_types, markers, palette):
        sub = merged_pca_df[merged_pca_df['FailureType'] == ft]
        ax_scatter.scatter(sub['PCA_ND_Params'], sub['PCA_Calibration'],
                           label=ft, marker=m, color=c, s=50,
                           edgecolor='k', linewidth=0.3)
    ax_scatter.set_xlabel('ND-Parameters PCA (1st component)')
    ax_scatter.set_ylabel('BW-Parameters PCA (1st component)')
    ax_scatter.grid(True)

    # Move legend outside the plot
    ax_scatter.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax_scatter.set_xlim(-4, 4)
    ax_scatter.set_ylim(-4, 4)

    # Histogram (PCA_ND_Params)
    ax_hist = plt.subplot(gs[1, 0], sharex=ax_scatter)
    grey_palette = sns.color_palette("Greys", n_colors=merged_pca_df['FailureType'].nunique())
    # add hatch patterns so stacked columns are visually distinct
    # tighten hatch spacing: decrease hatch line width and increase hatch character repetition
    plt.rcParams['hatch.linewidth'] = 0.6
    hatches = ['////', 'xxxx', 'oooo', '....', '****', '----', '++++', '||||', '\\\\\\\\']
    failure_types = merged_pca_df['FailureType'].unique()[::-1]
    for i, failure_type in enumerate(failure_types):
        color = grey_palette[::-1][i % len(grey_palette)]
        hatch = hatches[i % len(hatches)]
        subset = merged_pca_df[merged_pca_df['FailureType'] == failure_type]
        ax_hist.hist(
            subset['PCA_ND_Params'],
            bins=20,
            alpha=0.9,
            label=failure_type,
            stacked=True,
            color=color,
            edgecolor='white',
            linewidth=0.1,
            hatch=hatch
        )

    ax_hist.set_xlabel('ND-Parameters PCA (1st component)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True)

    # Move legend outside the histogram
    ax_hist.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.7)  # Adjust the right margin to make space for the legend

    # plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pca_scatter_histogram_plot.pdf'), bbox_inches='tight')
    plt.show()


## --------
## Plot # 5
## --------
error_metrics_path = 'gp_training_data/error_metrics'
plot_type = 'hist'

# Load error metrics data
# Load the error metrics data for model 0, 1, 2, 3, 4, 5, avg
error_metrics_0 = pd.read_csv(os.path.join(error_metrics_path, 'err_metrics_split_00.csv'))
# Filter to only keep metrics with column split = 0
error_metrics_0 = error_metrics_0[error_metrics_0['split'] == 0]

error_metrics_1 = pd.read_csv(os.path.join(error_metrics_path, 'err_metrics_split_01.csv'))
error_metrics_1 = error_metrics_1[error_metrics_1['split'] == 1]
error_metrics_2 = pd.read_csv(os.path.join(error_metrics_path, 'err_metrics_split_02.csv'))
error_metrics_2 = error_metrics_2[error_metrics_2['split'] == 2]

error_metrics_3 = pd.read_csv(os.path.join(error_metrics_path, 'err_metrics_split_03.csv'))
error_metrics_3 = error_metrics_3[error_metrics_3['split'] == 3]

error_metrics_4 = pd.read_csv(os.path.join(error_metrics_path, 'err_metrics_split_04.csv'))
error_metrics_4 = error_metrics_4[error_metrics_4['split'] == 4]

error_metrics_5 = pd.read_csv(os.path.join(error_metrics_path, 'err_metrics_split_05.csv'))
error_metrics_5 = error_metrics_5[error_metrics_5['split'] == 5]

error_metrics_a = pd.read_csv(os.path.join(error_metrics_path, 'err_metrics_no_split.csv'))

# Remove points with error > 0.5
'''error_metrics_0 = error_metrics_0[error_metrics_0['err_surr_exp'] < 0.3]
error_metrics_1 = error_metrics_1[error_metrics_1['err_surr_exp'] < 0.3]
error_metrics_2 = error_metrics_2[error_metrics_2['err_surr_exp'] < 0.3]
error_metrics_3 = error_metrics_3[error_metrics_3['err_surr_exp'] < 0.3]
error_metrics_4 = error_metrics_4[error_metrics_4['err_surr_exp'] < 0.3]
error_metrics_5 = error_metrics_5[error_metrics_5['err_surr_exp'] < 0.3]
error_metrics_a = error_metrics_a[error_metrics_a['err_surr_exp'] < 0.3]
'''

density = True
if 5 in do_plots:

    sel_err_metric = 'err_surr_exp'

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1, 'wspace': 0.1})

    base_color = 'blue'
    shades = sns.light_palette(base_color, n_colors=6, reverse=True)
    bins = np.linspace(0.0, 0.3, 30)

    if plot_type == 'kde':
        # Plot KDEs and CDFs for the first 6 models
        for data, label, color in zip(
                [error_metrics_0, error_metrics_1, error_metrics_2, error_metrics_3, error_metrics_4, error_metrics_5],
                ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
                shades):
            plot_kde(data[sel_err_metric], label, color, axes[0, 0], axes[1, 0])
        axes[0, 0].set_ylabel('KDE Density', fontsize=14)
    else:
        # Plot histograms and CDFs for the first 6 models (add hatching for grayscale-friendly output)
        hatches6 = ['////', '\\', 'xx', '..', 'oo', '++']
        for i, (data, label, color) in enumerate(zip(
                [error_metrics_0, error_metrics_1, error_metrics_2, error_metrics_3, error_metrics_4, error_metrics_5],
                ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
                shades)):
            axes[0, 0].hist(
                data[sel_err_metric], bins=bins, alpha=0.5, label=label, color=color,
                density=density, edgecolor='black', linewidth=0.5, hatch=hatches6[i % len(hatches6)]
            )
            sorted_data = np.sort(data[sel_err_metric])
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            axes[1, 0].plot(sorted_data, cdf, color=color, linewidth=2, label=label)
        axes[0, 0].set_ylabel('Density', fontsize=14)

    axes[0, 0].legend(fontsize=12, title_fontsize=14)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].set_xlim(0.0, 0.3)
    axes[0, 0].set_xticklabels([])

    axes[1, 0].set_xlabel('MAE GP/Exp', fontsize=14)
    axes[1, 0].set_ylabel('Norm. Cum. Density', fontsize=14)
    axes[1, 0].legend(fontsize=12, title_fontsize=14)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].set_xlim(0.0, 0.3)

    if plot_type == 'kde':
        plot_kde(error_metrics_a[sel_err_metric], 'Final GP', 'gray', axes[0, 1], axes[1, 1])
    else:
        axes[0, 1].hist(
            error_metrics_a[sel_err_metric], bins=bins, alpha=0.7, label='Final GP', color='gray',
            density=density, edgecolor='black', linewidth=0.5, hatch='////'
        )
        axes[0, 1].legend(fontsize=12, title_fontsize=14)
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        axes[0, 1].set_xlim(0.0, 0.3)
        axes[0, 1].set_yticklabels([])
        axes[0, 1].set_xticklabels([])

        sorted_data = np.sort(error_metrics_a[sel_err_metric])
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 1].plot(sorted_data, cdf, color='gray', linewidth=2, label='Final GP')

    axes[0, 1].legend(fontsize=12, title_fontsize=14)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].set_xlim(0.0, 0.3)
    axes[0, 1].set_yticklabels([])
    axes[0, 1].set_xticklabels([])

    axes[1, 1].set_xlabel('MAE GP/Exp', fontsize=14)
    axes[1, 1].legend(fontsize=12, title_fontsize=14)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].set_xlim(0.0, 0.3)
    axes[1, 1].set_yticklabels([])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'error_metrics_pdf_cdf.pdf'), bbox_inches='tight')
    plt.show()


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

    axes[1, 0].set_xlabel('MAE GP/Exp', fontsize=14)
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

    axes[1, 1].set_xlabel('MAE GP/Exp', fontsize=14)
    axes[1, 1].legend(fontsize=12, title_fontsize=14)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].set_xlim(0.0, 0.3)  # Set x-axis limits
    axes[1, 1].set_yticklabels([])  # Remove y tick labels

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), bbox_inches='tight')
    plt.show()


'''

## --------
## Plot # 7
## --------
# Repeat plot 5, but filter 'FalureMode' to 'Flexure'
err_metrics_0_flexure = error_metrics_0[error_metrics_0['FailureMode'] == 'Flexure']
err_metrics_1_flexure = error_metrics_1[error_metrics_1['FailureMode'] == 'Flexure']
err_metrics_2_flexure = error_metrics_2[error_metrics_2['FailureMode'] == 'Flexure']
err_metrics_3_flexure = error_metrics_3[error_metrics_3['FailureMode'] == 'Flexure']
err_metrics_4_flexure = error_metrics_4[error_metrics_4['FailureMode'] == 'Flexure']
err_metrics_5_flexure = error_metrics_5[error_metrics_5['FailureMode'] == 'Flexure']
err_metrics_a_flexure = error_metrics_a[error_metrics_a['FailureMode'] == 'Flexure']

if 7 in do_plots:

    sel_err_metric = 'err_surr_exp'

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1, 'wspace': 0.1})

    base_color = 'green'
    shades = sns.light_palette(base_color, n_colors=6, reverse=True)
    bins = np.linspace(0.0, 0.3, 30)

    if plot_type == 'kde':
        # Plot KDEs and CDFs for the first 6 models
        for data, label, color in zip(
                [err_metrics_0_flexure, err_metrics_1_flexure, err_metrics_2_flexure, err_metrics_3_flexure, err_metrics_4_flexure, err_metrics_5_flexure],
                ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
                shades):
            plot_kde(data[sel_err_metric], label, color, axes[0, 0], axes[1, 0])
        axes[0, 0].set_ylabel('KDE Density', fontsize=14)
    else:
        # Plot histograms and CDFs for the first 6 models (add hatching for grayscale-friendly output)
        hatches6 = ['////', '\\', 'xx', '..', 'oo', '++']
        for i, (data, label, color) in enumerate(zip(
                [err_metrics_0_flexure, err_metrics_1_flexure, err_metrics_2_flexure, err_metrics_3_flexure, err_metrics_4_flexure, err_metrics_5_flexure],
                ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
                shades)):
            axes[0, 0].hist(
                data[sel_err_metric], bins=bins, alpha=0.5, label=label, color=color,
                density=density, edgecolor='black', linewidth=0.5, hatch=hatches6[i % len(hatches6)]
            )
            sorted_data = np.sort(data[sel_err_metric])
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            axes[1, 0].plot(sorted_data, cdf, color=color, linewidth=2, label=label)
        axes[0, 0].set_ylabel('Density', fontsize=14)

    axes[0, 0].legend(fontsize=12, title_fontsize=14)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].set_xlim(0.0, 0.3)
    axes[0, 0].set_xticklabels([])

    axes[1, 0].set_xlabel('MAE GP/Exp', fontsize=14)
    axes[1, 0].set_ylabel('Norm. Cum. Density', fontsize=14)
    axes[1, 0].legend(fontsize=12, title_fontsize=14)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].set_xlim(0.0, 0.3)

    if plot_type == 'kde':
        plot_kde(err_metrics_a_flexure[sel_err_metric], 'Final GP', 'gray', axes[0, 1], axes[1, 1])
    else:
        axes[0, 1].hist(
            err_metrics_a_flexure[sel_err_metric], bins=bins, alpha=0.7, label='Final GP', color='gray',
            density=density, edgecolor='black', linewidth=0.5, hatch='////'
        )
        axes[0, 1].legend(fontsize=12, title_fontsize=14)
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        axes[0, 1].set_xlim(0.0, 0.3)
        axes[0, 1].set_yticklabels([])
        axes[0, 1].set_xticklabels([])

        sorted_data = np.sort(error_metrics_a[sel_err_metric])
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 1].plot(sorted_data, cdf, color='gray', linewidth=2, label='Final GP')

    axes[0, 1].legend(fontsize=12, title_fontsize=14)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].set_xlim(0.0, 0.3)
    axes[0, 1].set_yticklabels([])
    axes[0, 1].set_xticklabels([])

    axes[1, 1].set_xlabel('MAE GP/Exp', fontsize=14)
    axes[1, 1].legend(fontsize=12, title_fontsize=14)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].set_xlim(0.0, 0.3)
    axes[1, 1].set_yticklabels([])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'error_metrics_pdf_cdf_flexure.pdf'), bbox_inches='tight')
    plt.show()


# Do same plot as above for FailureTYpe = Shear or Flexure-Shear
# Repeat plot 5, but filter 'FailureMode' to 'Flexure-Shear' or 'Shear'
err_metrics_0_shear = error_metrics_0[error_metrics_0['FailureMode'].isin(['Flexure-Shear', 'Shear'])]
err_metrics_1_shear = error_metrics_1[error_metrics_1['FailureMode'].isin(['Flexure-Shear', 'Shear'])]
err_metrics_2_shear = error_metrics_2[error_metrics_2['FailureMode'].isin(['Flexure-Shear', 'Shear'])]
err_metrics_3_shear = error_metrics_3[error_metrics_3['FailureMode'].isin(['Flexure-Shear', 'Shear'])]
err_metrics_4_shear = error_metrics_4[error_metrics_4['FailureMode'].isin(['Flexure-Shear', 'Shear'])]
err_metrics_5_shear = error_metrics_5[error_metrics_5['FailureMode'].isin(['Flexure-Shear', 'Shear'])]
err_metrics_a_shear = error_metrics_a[error_metrics_a['FailureMode'].isin(['Flexure-Shear', 'Shear'])]


if 8 in do_plots:

    sel_err_metric = 'err_surr_exp'

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1, 'wspace': 0.1})

    base_color = 'red'
    shades = sns.light_palette(base_color, n_colors=6, reverse=True)
    bins = np.linspace(0.0, 0.3, 30)

    if plot_type == 'kde':
        # Plot KDEs and CDFs for the first 6 models
        for data, label, color in zip(
                [err_metrics_0_shear, err_metrics_1_shear, err_metrics_2_shear, err_metrics_3_shear, err_metrics_4_shear, err_metrics_5_shear],
                ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
                shades):
            plot_kde(data[sel_err_metric], label, color, axes[0, 0], axes[1, 0])
        axes[0, 0].set_ylabel('KDE Density', fontsize=14)
    else:
        # Plot histograms and CDFs for the first 6 models (add hatching for grayscale-friendly output)
        hatches6 = ['////', '\\', 'xx', '..', 'oo', '++']
        for i, (data, label, color) in enumerate(zip(
                [err_metrics_0_shear, err_metrics_1_shear, err_metrics_2_shear, err_metrics_3_shear, err_metrics_4_shear, err_metrics_5_shear],
                ['Val. Set 1', 'Val. Set 2', 'Val. Set 3', 'Val. Set 4', 'Val. Set 5', 'Val. Set 6'],
                shades)):
            axes[0, 0].hist(
                data[sel_err_metric], bins=bins, alpha=0.5, label=label, color=color,
                density=density, edgecolor='black', linewidth=0.5, hatch=hatches6[i % len(hatches6)]
            )
            sorted_data = np.sort(data[sel_err_metric])
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            axes[1, 0].plot(sorted_data, cdf, color=color, linewidth=2, label=label)
        axes[0, 0].set_ylabel('Density', fontsize=14)

    axes[0, 0].legend(fontsize=12, title_fontsize=14)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].set_xlim(0.0, 0.3)
    axes[0, 0].set_xticklabels([])

    axes[1, 0].set_xlabel('MAE GP/Exp', fontsize=14)
    axes[1, 0].set_ylabel('Norm. Cum. Density', fontsize=14)
    axes[1, 0].legend(fontsize=12, title_fontsize=14)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].set_xlim(0.0, 0.3)

    if plot_type == 'kde':
        plot_kde(err_metrics_a_shear[sel_err_metric], 'Final GP', 'gray', axes[0, 1], axes[1, 1])
    else:
        axes[0, 1].hist(
            err_metrics_a_shear[sel_err_metric], bins=bins, alpha=0.7, label='Final GP', color='gray',
            density=density, edgecolor='black', linewidth=0.5, hatch='////'
        )
        axes[0, 1].legend(fontsize=12, title_fontsize=14)
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        axes[0, 1].set_xlim(0.0, 0.3)
        axes[0, 1].set_yticklabels([])
        axes[0, 1].set_xticklabels([])

        sorted_data = np.sort(error_metrics_a[sel_err_metric])
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 1].plot(sorted_data, cdf, color='gray', linewidth=2, label='Final GP')

    axes[0, 1].legend(fontsize=12, title_fontsize=14)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].set_xlim(0.0, 0.3)
    axes[0, 1].set_yticklabels([])
    axes[0, 1].set_xticklabels([])

    axes[1, 1].set_xlabel('MAE GP/Exp', fontsize=14)
    axes[1, 1].legend(fontsize=12, title_fontsize=14)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].set_xlim(0.0, 0.3)
    axes[1, 1].set_yticklabels([])

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'error_metrics_pdf_cdf_shear.pdf'), bbox_inches='tight')
    plt.show()


density = False
if 9 in do_plots:
    # Plot for final models (shear and flexure and all)
    err_metric_1 = 'err_surr_cal'
    err_metric_2 = 'err_surr_exp'
    err_metric_3 = 'err_cal_exp'

    # The plot is the following: kde or hist (depending on selection) for a models shear and flexure separate on the left, and toghether on the right.
    # Second row of plots are the corresponding CDFs.
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1, 'wspace': 0.1})

    # base_color = 'red'
    # shades = sns.light_palette(base_color, n_colors=6, reverse=True)

    bins = np.linspace(0.0, 0.3, 30)
    alpha = 0.5
    if plot_type == 'kde':
        # First error metric (surrogave vs calibration)
        plot_kde(err_metrics_a_shear[err_metric_1], 'Shear GP', 'red', axes[0, 0], axes[1, 0])
        plot_kde(err_metrics_a_flexure[err_metric_1], 'Flexure GP', 'green', axes[0, 0], axes[1, 0])
        plot_kde(np.concatenate([err_metrics_a_shear[err_metric_1], err_metrics_a_flexure[err_metric_1]]), 'All GP', 'gray', axes[0, 0], axes[1, 0])

        # Second error metric (surrogate vs experiment)
        plot_kde(err_metrics_a_shear[err_metric_2], 'Shear GP', 'red', axes[0, 1], axes[1, 1])
        plot_kde(err_metrics_a_flexure[err_metric_2], 'Flexure GP', 'green', axes[0, 1], axes[1, 1])
        plot_kde(np.concatenate([err_metrics_a_shear[err_metric_2], err_metrics_a_flexure[err_metric_2]]), 'All GP', 'gray', axes[0, 1], axes[1, 1])

    else:
        # First error metric (surrogate vs calibration) — add hatch patterns and bring Shear to front
        hatch_shear, hatch_flexure, hatch_all = '////', '\\', '..'
        # Draw 'All' first, then Flexure, then Shear last with highest zorder
        axes[0, 0].hist(
            np.concatenate([err_metrics_a_shear[err_metric_1], err_metrics_a_flexure[err_metric_1]]), bins=bins, alpha=alpha,
            label='Final GP All', color='gray', density=density, edgecolor='black', linewidth=0.5, hatch=hatch_all, zorder=1
        )
        axes[0, 0].hist(
            err_metrics_a_flexure[err_metric_1], bins=bins, alpha=alpha, label='Final GP Flexure', color='green',
            density=density, edgecolor='black', linewidth=0.5, hatch=hatch_flexure, zorder=2
        )
        axes[0, 0].hist(
            err_metrics_a_shear[err_metric_1], bins=bins, alpha=alpha, label='Final GP Shear', color='red',
            density=density, edgecolor='black', linewidth=0.5, hatch=hatch_shear, zorder=3
        )
        
        # Second error metric (surrogate vs experiment) — add hatch patterns and bring Shear to front
        axes[0, 1].hist(
            np.concatenate([err_metrics_a_shear[err_metric_2], err_metrics_a_flexure[err_metric_2]]), bins=bins, alpha=alpha,
            label='Final GP All', color='gray', density=density, edgecolor='black', linewidth=0.5, hatch=hatch_all, zorder=1
        )
        axes[0, 1].hist(
            err_metrics_a_flexure[err_metric_2], bins=bins, alpha=alpha, label='Final GP Flexure', color='green',
            density=density, edgecolor='black', linewidth=0.5, hatch=hatch_flexure, zorder=2
        )
        axes[0, 1].hist(
            err_metrics_a_shear[err_metric_2], bins=bins, alpha=alpha, label='Final GP Shear', color='red',
            density=density, edgecolor='black', linewidth=0.5, hatch=hatch_shear, zorder=3
        )

        # Now, do cumulative distribution
        # Cumulative distribution (CDF) for surrogate vs calibration
        sorted_shear_1 = np.sort(err_metrics_a_shear[err_metric_1])
        cdf_shear_1 = np.arange(1, len(sorted_shear_1) + 1) / len(sorted_shear_1)
        axes[1, 0].plot(sorted_shear_1, cdf_shear_1, color='red', linewidth=2.5, linestyle='--', label='Final GP Shear', zorder=3)

        sorted_flexure_1 = np.sort(err_metrics_a_flexure[err_metric_1])
        cdf_flexure_1 = np.arange(1, len(sorted_flexure_1) + 1) / len(sorted_flexure_1)
        axes[1, 0].plot(sorted_flexure_1, cdf_flexure_1, color='green', linewidth=2, linestyle='-.', label='Final GP Flexure', zorder=2)
        sorted_all = np.sort(np.concatenate([err_metrics_a_shear[err_metric_1], err_metrics_a_flexure[err_metric_1]]))
        cdf_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
        axes[1, 0].plot(sorted_all, cdf_all, color='gray', linewidth=2, label='Final GP All Data', zorder=1)

        # Cumulative distribution (CDF) for surrogate vs calibration
        sorted_shear_2 = np.sort(err_metrics_a_shear[err_metric_2])
        cdf_shear_2 = np.arange(1, len(sorted_shear_2) + 1) / len(sorted_shear_2)
        axes[1, 1].plot(sorted_shear_2, cdf_shear_2, color='red', linewidth=2.5, linestyle='--', label='Final GP Shear', zorder=3)

        sorted_flexure_2 = np.sort(err_metrics_a_flexure[err_metric_2])
        cdf_flexure_2 = np.arange(1, len(sorted_flexure_2) + 1) / len(sorted_flexure_2)
        axes[1, 1].plot(sorted_flexure_2, cdf_flexure_2, color='green', linewidth=2, linestyle='-.', label='Final GP Flexure', zorder=2)
        sorted_all = np.sort(np.concatenate([err_metrics_a_shear[err_metric_2], err_metrics_a_flexure[err_metric_2]]))
        cdf_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
        axes[1, 1].plot(sorted_all, cdf_all, color='gray', linewidth=2, label='Final GP All Data', zorder=1)


    axes[0, 0].set_ylabel('Number of tests', fontsize=14)
    axes[0, 0].legend(fontsize=12, title_fontsize=12)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].set_xlim(0.0, 0.3)
    axes[0, 0].set_xticklabels([])  # Remove x tick labels

    axes[0, 1].legend(fontsize=12, title_fontsize=12)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].set_xlim(0.0, 0.3)
    axes[0, 1].set_xticklabels([])  # Remove x tick labels
    axes[0, 1].set_yticklabels([])  # Remove y tick labels

    axes[1, 0].set_xlabel('MAE GP/Cal', fontsize=14)
    axes[1, 0].set_ylabel('Cumulative Density', fontsize=14)
    axes[1, 0].legend(fontsize=12, title_fontsize=14)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    axes[1, 0].set_xlim(0.0, 0.3)

    axes[1, 1].set_xlabel('MAE GP/Exp', fontsize=14)
    axes[1, 1].legend(fontsize=12, title_fontsize=14)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    axes[1, 1].set_xlim(0.0, 0.3)
    axes[1, 1].set_yticklabels([])  # Remove y tick labels

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'error_metrics_full_model.pdf'), bbox_inches='tight')
    plt.show()


# Selected hysteresis plots

# Create a folder for the selected figures. If folder already exists, clean it up.
selected_figures_observed = os.path.join(figures_dir, 'sel_hyst_observed')
if not os.path.exists(selected_figures_observed):
    os.makedirs(selected_figures_observed)
else:
    # If folder exists, clean it up
    for filename in os.listdir(selected_figures_observed):
        file_path = os.path.join(selected_figures_observed, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

selected_figures_unobserved = os.path.join(figures_dir, 'sel_hyst_unobserved')
if not os.path.exists(selected_figures_unobserved):
    os.makedirs(selected_figures_unobserved)
else:
    # If folder exists, clean it up
    for filename in os.listdir(selected_figures_unobserved):
        file_path = os.path.join(selected_figures_unobserved, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


sel_ids = [2, 5, 7, 291, 22, 24, 228, 273, 137, 49, 116, 215]

# Load the data_split.csv files from gp_training_data/processed/gpModelFlexure and gpModelShear
data_flexure = pd.read_csv(os.path.join('gp_training_data', 'processed', 'gpModelFlexure', 'data_split.csv'))
data_shear = pd.read_csv(os.path.join('gp_training_data', 'processed', 'gpModelShear', 'data_split.csv'))
# Merge (concatenate)
data_split = pd.concat([data_flexure, data_shear], ignore_index=True)

# Check the length of data_split
print(f"Length of data_split: {len(data_split)}")

for id in sel_ids:
    # Create file name
    UniqueId = str(id).zfill(3)
    filename = f'UniqueId_{UniqueId}.pdf'

    # ::: Observed data :::
    # Move the filename from the 'Figures/surrogate_hysteresis/no_split' to the selected figures directory
    src_path = os.path.join(figures_dir, 'surrogate_hysteresis', 'no_split', filename)
    if os.path.isfile(src_path):
        shutil.copy(src_path, selected_figures_observed)
        print(f"Copied {src_path} to {selected_figures_observed}")
    else:
        print(f"File not found: {src_path}")

    # ::: Unobserved data :::
    # Now, find the id in the UniqueId column in data_split, and extract the value in the split column
    split_value = data_split.loc[data_split['UniqueId'] == id, 'split']
    if not split_value.empty:
        print(f"Found split value for {UniqueId}: {split_value.values[0]}")
    else:
        print(f"No split value found for {UniqueId}")

    split_name = f'split_{str(split_value.values[0]).zfill(2)}'
    # Now, go to the 'Figures/surrogate_hysteresis/{split_name} to get the selected figures and place them in the selected figures directory

    src_dir = os.path.join(figures_dir, 'surrogate_hysteresis', split_name)
    if os.path.isdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy(src_file, selected_figures_unobserved)
            print(f"Copied {src_file} to {selected_figures_unobserved}")
        else:
            print(f"File not found: {src_file}")

