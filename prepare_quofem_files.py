import os
import shutil

test_files_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\test_data'
model_files_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model\column_model'
cal_file_dir = r'C:\Users\Miguel.MIGUEL-DESK\Documents\GitHub\RC_Column_Model'
test_id = 269

# Copy the test file
shutil.copyfile(os.path.join(test_files_dir, f'test_{test_id}.json'), os.path.join(model_files_dir, 'test_file.json'))

# Copy the calibration file
shutil.copyfile(os.path.join(test_files_dir, f'cal_{test_id}.csv'), os.path.join(cal_file_dir, 'calibration_file.csv'))
shutil.copyfile(os.path.join(test_files_dir, f'cal_{test_id}.csv'), os.path.join(model_files_dir, 'cal_file.csv'))

