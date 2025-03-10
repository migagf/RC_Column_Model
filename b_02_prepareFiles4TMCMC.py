# This code prepares the test and calibration files for TMCMC using quoFEM
import os
import shutil
import pandas as pd

# Set directory containing the test data
os.getcwd()

# Get current folder
current_folder = os.getcwd()

# Folder with the JSON files with test data
json_dir = current_folder + '/test_data/'

# Folder with the model files
model_files_dir = current_folder + '/column_model/'

# Load the database
data = pd.read_csv('merged_data.csv')

# For each curve
iniii = 2
maxii = 26 # len(data) (have only processed 15 of the curves)

for ii in range(iniii, maxii+1):
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
