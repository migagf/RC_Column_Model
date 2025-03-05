# This code prepares the test and calibration files for calibration
import os
import shutil

# Set directory containing the test data
os.getcwd()

dir_to_test_files = os.path.join(os.getcwd(), 'test_data')
#test_files_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data'

# Process the test data to create a calibration file

model_files_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\column_model'

# model_files_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model'
cal_file_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model'

# Last running 270
test_id = 270

# Copy the test file
shutil.copyfile(os.path.join(test_files_dir, f'test_{test_id}.json'), os.path.join(model_files_dir, 'test_file.json'))

# Copy the calibration file
shutil.copyfile(os.path.join(test_files_dir, f'cal_{test_id}.csv'), os.path.join(cal_file_dir, 'calibration_file.csv'))
shutil.copyfile(os.path.join(test_files_dir, f'cal_{test_id}.csv'), os.path.join(model_files_dir, 'cal_file.csv'))

