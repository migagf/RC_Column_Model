# Compile quoFEM calibrations

# Process the calibration data from the downloaded files

import os
import zipfile
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pprint
import traceback
import re

# Use latex for plots (optional, can be commented out)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

'''
index   parameter
2       gamma
3       kappa
4       eta1
5       sig
6       lam
7       mup
8       sigp
9       rsmax
10      alpha
11      alpha1
12      alpha2
13      betam1
14      n
15      kappa_k
'''

def get_info(test_dir, log_file):
    '''
    Get information of the calibration
    This function explores the test_dir and checks whether the application corresponds
    to a UCSD-UQ calibration. If it is, then if extracts the information from the calibration.
    '''
    # Before trying anything, check the tapisjob.out file. If the last line of the file contains utils then proceed. Otherwise, print "failed analysis" in the log_file
    tapisjobFile = test_dir + r'/tapisjob.out'
    try:
        with open(tapisjobFile) as f:
            lines = f.readlines()
            if not any("utils" in line for line in lines[-5:]):
                write_to_log(log_file, 'failed analysis')
                return [], [], [], [], 0
    except Exception as e:
        write_to_log(log_file, f'failed analysis ...tapisjob.out not found... (exception: {e})')
        return [], [], [], [], 0

    # (1) Unzip the results.zip file
    resultsFileDir = test_dir + '/results'
    resultsZip = test_dir + '/results.zip'

    if not os.path.exists(resultsFileDir):
        write_to_log(log_file, 'Working on folder ' + test_dir)
        write_to_log(log_file, 'Results folder does not exist... Trying to unzip results.zip')
        if os.path.exists(resultsZip):
            write_to_log(log_file, 'Unzipping results.zip')
            with zipfile.ZipFile(resultsZip, 'r') as zip_ref:
                zip_ref.extractall(test_dir)
        else:
            write_to_log(log_file, 'results.zip file does not exist')
            check = 0

    # (2) Load the scInput file
    # Try to open the scInput.json file...
    try:
        scInputFile = test_dir + r'/results/templatedir/scInput.json'
        with open(scInputFile) as f:
            scInput = json.load(f)
        check = 1
    except Exception as e:
        write_to_log(log_file, f'Error trying to open scInput.json file... Cause: {e}')
        scInputFile = test_dir + r'/DS_Input_Run/templatedir/scInput.json'
        check = 0

    # (3) Check if UQ application is UCSD-UQ. If it's not, exit the function
    if check == 1:
        if scInput["Applications"]['UQ']['Application'] != 'UCSD-UQ':
            write_to_log(log_file, f"Found UQ application: {scInput['Applications']['UQ']['Application']}")
            write_to_log(log_file, 'Not a UCSD-UQ Calibration')
            return [], [], [], [], 0
    
        else:
            try:
                # If it is a UCSD-UQ calibration, continue with the process...

                # (4) Unzip the templatedir.zip file
                templatedirDir = test_dir + '/templatedir'
                templatedirZip = test_dir + '/templatedir.zip'

                if not os.path.exists(templatedirDir):
                    write_to_log(log_file, 'Template folder does not exist... Trying to unzip templatedir.zip')
                    if os.path.exists(templatedirZip):
                        write_to_log(log_file, 'Unzipping templatedir.zip')
                        with zipfile.ZipFile(templatedirZip, 'r') as zip_ref:
                            zip_ref.extractall(test_dir)
                    else:
                        write_to_log(log_file, 'templatedir.zip file does not exist')

                # (5) Load the dakotaTab.out file
                dakotaTabFile = test_dir + r'/results/dakotaTab.out'
                results = pd.read_csv(dakotaTabFile, sep='\s+')

                # (6) Load the test file
                testFile = templatedirDir + r'/test_file.json'
                with open(testFile) as f:
                    test_file = json.load(f)

                # Section 3: Get the info
                # Split the test_dir by the backslash
                test_dir_split = test_dir.split('\\')
                # Get the last part of the split
                test_dir_name = test_dir_split[-1]

                UniqueId = scInput["Applications"]["FEM"]["ApplicationData"]["MS_Path"][-3:]

                write_to_log(log_file, f'\n ::: Found test with UniqueId: {UniqueId}')
                write_to_log(log_file, f'Location: {test_dir_name} \n :::')

                info = {
                    'UniqueId': UniqueId,
                    'Name': test_file['Name'],
                    'Location': test_dir_name
                    }

                return info, scInput, test_file, results, 1
            
            except Exception as e:
                write_to_log(log_file, f'Error trying to get calibration data... Cause: {e}')
                return [], [], [], [], 0
    else:
        return [], [], [], [], 0


def plot_hysteresis(test_file, results, filename, info, output_dir, save=False):

    length = test_file['L_Inflection']
    peak_force = np.max(np.array(test_file['cal_data']['force']))
    mae = info['res_mean']

    plt.figure(figsize=(3, 3))
    random_indices = np.random.choice(range(0, 200), size=30, replace=False)
    for ii in random_indices:
        if ii == random_indices[0]: # Plot so that legend shows up
            plt.plot(np.array(test_file['cal_data']['disp'])/length, 
                     np.array(results.iloc[ii, 17::])/peak_force, 
                     'r:.', linewidth=0.3, alpha=0.3, markersize=2.0, label='Post. Samples')
        else:
            plt.plot(np.array(test_file['cal_data']['disp'])/length, 
                    np.array(results.iloc[ii, 17::])/peak_force, 
                    'r:.', linewidth=0.3, alpha=0.3, markersize=2.0, label=None)

    plt.plot(np.array(test_file['cal_data']['disp'])/length, 
             np.array(test_file['cal_data']['force'])/peak_force, 
             'b-.', alpha=0.8, label='Exp.', linewidth=1.0, markersize=3.0)
    
    
    # Show only text up to the year number
    full_name = str(test_file.get('Name', ''))
    m = re.search(r'\b(\d{4})\b', full_name)
    if m:
        name_short = full_name[:m.end()].strip()
    else:
        name_short = full_name.split(',')[0].strip()
    title = name_short + '\n | PEER ID: ' + filename[-3:]
    plt.title(title, fontsize=8)
    plt.xlabel('Drift Ratio $\Delta/h$')
    plt.ylabel('Normalized Shear $V/V_s$')
    plt.legend(loc='upper left')
    # Add text in bottom right corner with the MAE
    plt.text(0.98, 0.02, r'MAE = %.4f' % mae,
             horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gca().transAxes,
             fontsize=10, color='black',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    plt.xticks(ticks=plt.xticks()[0], labels=[f'{x:.2f}' for x in plt.xticks()[0]])
    plt.tight_layout()
    
    if save:
        # Save high-quality PNG instead of PDF for better raster output
        out_path = os.path.join(output_dir, 'plots')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        #plt.savefig(os.path.join(out_path, f'{filename}.png'), dpi=600, bbox_inches='tight')
        # Save as PDF
        plt.savefig(os.path.join(out_path, f'{filename}.pdf'), bbox_inches='tight', facecolor='white')


def get_residual_info(test_file, results):
    
    # Get statistics on the residuals
    residuals = np.array(results.iloc[:, 17::]) - np.array(test_file['cal_data']['force'])
    residuals = np.abs(residuals / np.max(np.array(test_file['cal_data']['force'])))

    # get mean residual about axis 0
    residuals = np.mean(residuals, axis=1)

    # Mean, std, max, min of mean of residuals
    mean = np.mean(residuals)
    median = np.median(residuals)
    std = np.std(residuals)
    max = np.max(residuals)
    min = np.min(residuals)

    residual_statistics = {'mean': mean, 'median': median, 'std': std, 'max': max, 'min': min}
    
    # Get the best fit index
    best_fit_index = np.argmin(residuals)

    # Get parameters for best fit
    best_fit_parameters = results.iloc[best_fit_index, 2:16].values
    
    return residual_statistics, best_fit_parameters


def write_to_log(log_file, message):
    with open(log_file, 'a') as logf:
        logf.write(f"{message}\n")
    print(message)


if __name__ == '__main__':

    # The directory where the files are stored
    remoteWorkDir = r'D:\tacc scratch'
    months = ['25_01', '25_02', '25_03', '25_08', '25_09']
    output_folder = r'gp_training_data\calibrations'
    figures_dir = r'gp_training_data\calibrations\plots'

    for month in months:
        # list folders in month directory to process each day
        days = os.listdir(os.path.join(remoteWorkDir, month))
        
        for day in days:
            # Logfile (to be stored in the day folder)
            log_file = os.path.join(remoteWorkDir, month, day, 'log.txt')
            # Remove the log file, if it exists
            if os.path.exists(log_file):
                os.remove(log_file)

            allJobs = os.listdir(os.path.join(remoteWorkDir, month, day))
            # Sort the list of allJobs by filename
            allJobs.sort()

            write_to_log(log_file, f"{len(allJobs)} jobs found...")

            for ii in range(0, len(allJobs)):
                write_to_log(log_file, f"\n\nProcessing job {ii+1} of {len(allJobs)}: {allJobs[ii]}")
                # Print the location of the file being processed
                write_to_log(log_file, f"Location: {os.path.join(remoteWorkDir, month, day, allJobs[ii])}")

                info, scInput, test_file, results, exitcode = get_info(os.path.join(remoteWorkDir, month, day, allJobs[ii]), log_file)
                try:
                    res_stats, best_fit = get_residual_info(test_file, results)
                except Exception as e:
                    write_to_log(log_file, f'Error trying to get residuals... Cause: {e}')
                    traceback.print_exc()
                    continue
                
                # Add fit information to the info dictionary
                info['res_mean'] = res_stats['mean']
                info['res_median'] = res_stats['median']
                info['res_std'] = res_stats['std']
                info['res_max'] = res_stats['max']
                info['res_min'] = res_stats['min']
                print(best_fit)
                
                (info['gamma'], info['kappa'], info['eta1'], info['sig'], info['lam'], 
                info['mup'], info['sigp'], info['rsmax'], info['alpha'], info['alpha1'], 
                info['alpha2'], info['betam1'], info['n'], info['kappa_k']) = best_fit.tolist()

                # Save the info dictionary to a dataframe
                info_df = pd.DataFrame(info, index=[0])
                print(info_df)

                # Open the calibration_info.csv file and check if the UniqueId exists
                if os.path.exists(os.path.join(output_folder, 'calibration_info.csv')):
                    calibration_info = pd.read_csv(os.path.join(output_folder, 'calibration_info.csv'))
                    if int(info['UniqueId']) in calibration_info['UniqueId'].values:
                        print('UniqueId already exists... ')
                        # Check if the existing Location is the same as the new one
                        existing_location = calibration_info.loc[calibration_info['UniqueId'] == int(info['UniqueId']), 'Location'].values[0]
                        if existing_location == info['Location']:
                            print('Location is the same... Updating the info')
                            # Replace the entire row with the new info
                            print(calibration_info.loc[calibration_info['UniqueId'] == int(info['UniqueId'])])
                            for col in info_df.columns:
                                calibration_info.loc[calibration_info['UniqueId'] == int(info['UniqueId']), col] = info_df.iloc[0][col]

                            # Plot and save
                            plot_hysteresis(test_file, results, 'test_' + info['UniqueId'] , info, output_dir=output_folder, save=True)

                            # Update the calibration_info.csv file
                            if os.path.exists(os.path.join(output_folder, 'calibration_info.csv')):
                                # Remove the existing file
                                os.remove(os.path.join(output_folder, 'calibration_info.csv'))

                            calibration_info.to_csv(os.path.join(output_folder, 'calibration_info.csv'), index=False)

                        else:
                            # In case the location is different, a new calibration was made
                            print('Location is different... ')

                            # Check the mean residuals
                            existing_res_mean = calibration_info.loc[calibration_info['UniqueId'] == int(info['UniqueId']), 'res_mean'].values[0]
                            if info['res_mean'] <= existing_res_mean:
                                print('New mean residual is less than the existing one... Updating the info')

                                # Replace the row with the new info (assign per-column to avoid shape mismatches)
                                print(calibration_info.loc[calibration_info['UniqueId'] == int(info['UniqueId'])])
                                pprint.pprint(info)
                                for col in info_df.columns:
                                    calibration_info.loc[calibration_info['UniqueId'] == int(info['UniqueId']), col] = info_df.iloc[0][col]

                                # Plot and save
                                plot_hysteresis(test_file, results, 'test_' + info['UniqueId'] , info, output_dir=output_folder, save=True)

                                if os.path.exists(os.path.join(output_folder, 'calibration_info.csv')):
                                    # Remove the existing file
                                    os.remove(os.path.join(output_folder, 'calibration_info.csv'))

                                calibration_info.to_csv(os.path.join(output_folder, 'calibration_info.csv'), index=False)

                                #
                            else:
                                # Existing mean residual is less than the new one, don't update the info
                                print('New mean residual is greater than the existing one...')
                                print('Existing mean residual: ', existing_res_mean)
                                print('New mean residual: ', info['res_mean'])
                        
                    else:
                        # Append the info to the csv file
                        print('UniqueId does not exist... Adding new info')
                        info_df = pd.DataFrame(info, index=[0])
                        info_df.to_csv(os.path.join(output_folder, 'calibration_info.csv'), mode='a', header=False, index=False)
                        plot_hysteresis(test_file, results, 'test_' + info['UniqueId'], info, output_dir=output_folder, save=True)
                else:
                    # Create the csv file and add the info as the first row
                    print('calibration_info.csv not found... Creating a new one')
                    calibration_info = pd.DataFrame(columns=info.keys())
                    calibration_info.to_csv(os.path.join(output_folder, 'calibration_info.csv'), index=False)
                    plot_hysteresis(test_file, results, 'test_' + info['UniqueId'] , info, output_dir=output_folder, save=True)

