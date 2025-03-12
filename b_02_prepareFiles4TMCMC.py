# This code prepares the test and calibration files for TMCMC using quoFEM
import os
import shutil
import pandas as pd
import json

# Set directory containing the test data
# os.getcwd()

def prepare_files_for_TMCMC(current_folder, json_dir, data, ii):
        
    # Get current folder
    # current_folder = os.getcwd()

    # Folder with the JSON files with test data
    # json_dir = current_folder + '/test_data/'

    # Folder with the model files
    model_files_dir = current_folder + '/column_model/'

    # Load the database
    # data = pd.read_csv('merged_data.csv')

    # For each curve
    # iniii = 2
    # maxii = 26 # len(data) (have only processed 15 of the curves)

    # selii = 1
    # for ii in range(selii, selii+1):
    print('Saving files for test ' + str(ii))
    test_id = str(int(data.UniqueId[ii])).zfill(3)
    folder_name = 'test_' + test_id

    # Create a folder for the calibration files
    model_files_dir = os.path.join(current_folder, 'quoFEM_TMCMC', folder_name)
    if not os.path.exists(model_files_dir):
        os.makedirs(model_files_dir)
    else:
        shutil.rmtree(model_files_dir)
        os.makedirs(model_files_dir)

    # Copy the test file
    shutil.copyfile(os.path.join(json_dir, f'test_{test_id}.json'), os.path.join(model_files_dir, 'test_file.json'))

    # Copy the calibration file
    shutil.copyfile(os.path.join(json_dir, f'cal_{test_id}.csv'), os.path.join(model_files_dir, 'cal_file.csv'))

    # Copy all files in the colum_model folder into the calibration folder
    for file in os.listdir(os.path.join(current_folder, 'column_model')):
        if file.endswith('.py'):
            src_file = os.path.join(current_folder, 'column_model', file)
            dest_file = os.path.join(model_files_dir, file)
            shutil.copyfile(src_file, dest_file)

    print('Done saving files')
    print('===')


def create_calibrations_log(destination, data):
    '''
    Call this function to create the calibrations log file from scratch
    '''
    # Create a DataFrame for the test matrix
    calibrations_log_df = pd.DataFrame(columns=['calId', 'UniqueId', 'npts', 'folder'])

    # Path to location of the test_files
    cwd = os.getcwd()
    filesdir = os.path.join(cwd, 'quoFEM_TMCMC')

    # Get folder names in filesdir
    folders = [f for f in os.listdir(filesdir) if os.path.isdir(os.path.join(filesdir, f))]

    # Loop over folders. If folder name starts with 'test_', get the test id
    for folder in folders:
        if folder.startswith('test_'):
            # UniqueId is the test id
            UniqueId = folder.split('_')[1]

            # CalId is the index of the test in the data DataFrame
            calId = str(data[data['UniqueId'] == int(UniqueId)].index[0]).zfill(3)
            
            # Load test_file.json to get npts
            test_file_path = os.path.join(filesdir, folder, 'test_file.json')
            with open(test_file_path, 'r') as f:
                test_data = json.load(f)
            npts = test_data['cal_data']['npts']

            # Append to DataFrame
            new_row = pd.DataFrame({'calId': [calId], 'UniqueId': [UniqueId], 'npts': [npts], 'folder': [folder]})
            calibrations_log_df = pd.concat([calibrations_log_df, new_row], ignore_index=True)
    
    # Sort dataframe by calId
    calibrations_log_df.sort_values(by='calId', inplace=True)

    # Save DataFrame to CSV
    calibrations_log_df.to_csv(destination, index=False)


def update_calibrations_log(current_folder, data, ii):
    '''
    This function creates a test matrix, storing the test Id, the number of points, and the filenames
    '''
    destination = os.path.join(current_folder, 'quoFEM_TMCMC', 'calibrations_log.csv')

    # If file doesn't exist, create it
    if not os.path.exists(destination):
        create_calibrations_log(destination, data)

    else:
        # Open the existing file
        calibrations_log_df = pd.read_csv(destination, dtype={'calId': str, 'UniqueId': str, 'npts': int, 'folder': str})
        
        # Check if the ii is part of the calId's in the calibrations file
        calId = str(ii).zfill(3)
        if calId not in calibrations_log_df['calId'].values:
            try:
                # If not, add a new entry
                UniqueId = str(int(data.UniqueId[ii])).zfill(3)

                # Load test_file.json to get npts
                test_file_path = os.path.join(current_folder, 'quoFEM_TMCMC', f'test_{UniqueId}', 'test_file.json')
                with open(test_file_path, 'r') as f:
                    test_data = json.load(f)
                npts = test_data['cal_data']['npts']

                # Append to DataFrame
                new_row = pd.DataFrame({'calId': [calId], 'UniqueId': [UniqueId], 'npts': [npts], 'folder': [f'test_{UniqueId}']})
                calibrations_log_df = pd.concat([calibrations_log_df, new_row], ignore_index=True)
                print(calibrations_log_df)
                # Sort dataframe by calId
                calibrations_log_df.sort_values(by='calId', inplace=True)

                # Save DataFrame to CSV
                calibrations_log_df.to_csv(destination, index=False)
            except Exception as que_paso:
                print(f'Error processing test {ii}. Possibly missing test_ folder \n', que_paso)
        else:
            UniqueId = str(int(data.UniqueId[ii])).zfill(3)
            # Print a message if the entry already exists
            print(f'Entry for calId {calId} already exists in the log., updating')
            test_file_path = os.path.join(current_folder, 'quoFEM_TMCMC', f'test_{UniqueId}', 'test_file.json')
            with open(test_file_path, 'r') as f:
                test_data = json.load(f)
                        
            npts = test_data['cal_data']['npts']
            calibrations_log_df.loc[calibrations_log_df['calId'] == calId, 'npts'] = npts
            calibrations_log_df.to_csv(destination, index=False)
    pass


# Example usage
if __name__ == "__main__":
    # Example current folder and data
    current_folder = os.getcwd()
    
    data = pd.read_csv(os.path.join(current_folder, 'merged_data.csv'))

    # Example test index
    update_calibrations_log(current_folder, data, 10)