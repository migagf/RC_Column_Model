# Compile quoFEM calibrations

# Process the calibration data from the downloaded files

import os
import zipfile
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def get_info(test_dir):

    # (1) Unzip the results.zip file
    resultsFileDir = test_dir + '/results'
    resultsZip = test_dir + '/results.zip'

    if not os.path.exists(resultsFileDir):
        print('Results folder does not exist... Trying to unzip results.zip')
        if os.path.exists(resultsZip):
            print('Unzipping results.zip')
            with zipfile.ZipFile(resultsZip, 'r') as zip_ref:
                zip_ref.extractall(test_dir)
        else:
            print('results.zip file does not exist')
            check = 0

    # (2) Load the scInput file
    # Try to open the scInput.json file...
    try:
        scInputFile = test_dir + r'/results/templatedir/scInput.json'
        with open(scInputFile) as f:
            scInput = json.load(f)
        check = 1
    except Exception as e:
        print('Error trying to open scInput.json file... ', e)
        check = 0

    # (3) Check if UQ application is UCSD-UQ. If it's not, exit the function
    if check == 1:
        if scInput["Applications"]['UQ']['Application'] != 'UCSD-UQ':
            print(scInput["Applications"]['UQ'])
            print('Not a UCSD-UQ Calibration')
            return [], [], [], []
    
        else:
            try:
                # If it is a UCSD-UQ calibration, continue with the process...

                # (4) Unzip the templatedir.zip file
                templatedirDir = test_dir + '/templatedir'
                templatedirZip = test_dir + '/templatedir.zip'

                if not os.path.exists(templatedirDir):
                    print('Template folder does not exist... Trying to unzip templatedir.zip')
                    if os.path.exists(templatedirZip):
                        print('Unzipping templatedir.zip')
                        with zipfile.ZipFile(templatedirZip, 'r') as zip_ref:
                            zip_ref.extractall(test_dir)
                    else:
                        print('templatedir.zip file does not exist')
                
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
                info = {'UniqueId': UniqueId,
                        'Name': test_file['Name'],
                        'Location': test_dir_name
                        }

                return info, scInput, test_file, results
            except Exception as e:
                print('Error trying to get calibration data... ', e)
                return [], [], [], []
    else:
        return [], [], [], []


def plot_hysteresis(test_file, results, filename, save=False):

    plt.figure()
    for ii in range(0, 200):
        plt.plot(test_file['cal_data']['disp'], results.iloc[ii, 17::], 'r:.', linewidth=0.1, alpha=0.2)

    plt.plot(test_file['cal_data']['disp'], test_file['cal_data']['force'], 'b:.', alpha=0.5)
    plt.title(filename)

    if save:
        plt.savefig('CalibrationPlots/'+filename+'.pdf')
    
    # plt.show()


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


if __name__ == '__main__':

    # The directory where the files are stored
    remoteWorkDir = r'D:\tacc scratch'
    month = '25_03'
    day = '17'

    allJobs = os.listdir(os.path.join(remoteWorkDir, month, day))

    for ii in range(0, len(allJobs)):
        info, scInput, test_file, results = get_info(os.path.join(remoteWorkDir, month, day, allJobs[ii]))
        try:
            res_stats, best_fit = get_residual_info(test_file, results)
        except Exception as e:
            print('Error trying to get residuals... ', e)
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

        #
        print(info)

        # Save the info dictionary to a csv file

        # Open the calibration_info.csv file and check if the UniqueId exists
        if os.path.exists('calibration_info.csv'):
            calibration_info = pd.read_csv('calibration_info.csv')
            if int(info['UniqueId']) in calibration_info['UniqueId'].values:
                print('UniqueId already exists... ')
                # Check if the existing Location is the same as the new one
                existing_location = calibration_info.loc[calibration_info['UniqueId'] == int(info['UniqueId']), 'Location'].values[0]
                if existing_location == info['Location']:
                    print('Location is the same... Updating the info')
                    # Replace the row with the new info
                    calibration_info = calibration_info.replace(info['UniqueId'], info)

                    # Plot and save
                    plot_hysteresis(test_file, results, 'test_' + info['UniqueId'] , save=True)
                    calibration_info.to_csv('calibration_info.csv', index=False)
                else:
                    print('Location is different... ')

                    # Check the mean residuals
                    existing_res_mean = calibration_info.loc[calibration_info['UniqueId'] == int(info['UniqueId']), 'res_mean'].values[0]
                    if info['res_mean'] <= existing_res_mean:
                        print('New mean residual is less than the existing one... Updating the info')
                        # Replace the row with the new info

                        calibration_info = calibration_info.replace(info['UniqueId'], info)
                        # Plot and save
                        plot_hysteresis(test_file, results, 'test_' + info['UniqueId'] , save=True)
                        calibration_info.to_csv('calibration_info.csv', index=False)

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
                info_df.to_csv('calibration_info.csv', mode='a', header=False, index=False)
                plot_hysteresis(test_file, results, 'test_' + info['UniqueId'] , save=True)
        else:
            # Create the csv file and add the info as the first row
            print('calibration_info.csv not found... Creating a new one')
            calibration_info = pd.DataFrame(columns=info.keys())
            calibration_info.to_csv('calibration_info.csv', index=False)
            plot_hysteresis(test_file, results, 'test_' + info['UniqueId'] , save=True)


