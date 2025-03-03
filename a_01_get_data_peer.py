#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code retrieves data from the PEER Performance Database
Gets the information from the website and puts it into convenient json files
for post processing

Created on Tue May 21 08:33:47 2024

@author: miguelgomez @ UC Berkeley
"""

# Extract text from PEER website

import requests
from bs4 import BeautifulSoup
import re
import json
import os


def find_float(text):
    # Define a regular expression pattern to match floats
    #pattern = r'[-+]?[0-9],?[0-9]*/.?[0-9]+'
    pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
    # Use re.findall to find all occurrences of floats in the text
    floats = re.findall(pattern, text)
    
    # Convert the strings to float values
    floats = [float(f.replace(',', '')) for f in floats]

    return floats


def get_table(url):
    # Step 1: Make an HTTP request to the webpage
    response = requests.get(url)
    
    # Step 2: Get the HTML content of the webpage
    html_content = response.content

    # Step 3: Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Step 4: Navigate and extract data:
    my_var = []
    # Example: Find all paragraphs and print their text
    paragraphs = soup.find_all('tr')
    hyperlinks = soup.find_all('a')
    
    for paragraph in paragraphs:
        my_var.append(paragraph.text)
    
    hlks = []
    for hyperlink in hyperlinks:
        #print(hyperlink['href'])
        hlks.append((hyperlink.text, hyperlink['href']))
    hlks_dict = dict(hlks)
    
    try:
        data_link = hlks_dict['Force Displacement Data (data)']
        print(data_link)
    except:
        print('No link found')
        data_link = 'none'
        
    return my_var, data_link


def get_force_deformation(url):
    
    # Step 1: Make an HTTP request to the webpage
    response = requests.get(url)
    text_content = response.text
    
    # Initialize two lists to store the columns
    column1 = []
    column2 = []
    
    # Split the content by lines
    lines = text_content.splitlines()
    
    # Process each line
    for line in lines:
        # Split the line by spaces (assuming only one space separates the columns)
        parts = line.split()
        
        # Append the data to the respective lists
        if len(parts) == 2:  # Ensure there are exactly two columns
            try:
                column1.append(float(parts[0]))
                column2.append(float(parts[1]))
            except:
                print('Skipping line', line)
    
    fd_data = {'disp': column1,
               'force': column2}
    
    return ('data', fd_data)


def get_test_data(text_vec, data_url):
    '''
    get_test_data
    
    Parameters
    ----------
    text_vec : LIST
        DESCRIPTION.

    Returns
    -------
    dict_with_data : DICT
        DESCRIPTION.

    '''
    # Empty list to fill with fields
    my_list = []
    
    for vals in text_vec:
        # Find Name
        if vals.find('Name') == 0:
            value = vals[6::]
            entry = ('Name', value)
            my_list.append(entry)
        
        # Find Type
        elif vals.find('Type: ') == 0:
            value = vals[6::]
            entry = ('Type', value)
            my_list.append(entry)
        
        # Find subtype for rect column
        elif vals.find('Type:R') == 0:
            value = vals[5::]
            entry = ('SubType', value)
            my_list.append(entry)
        
        # Find failure type
        elif vals.find('Failure Type') == 0:
            value = vals[13::]
            entry = ('FailureType', value)
            my_list.append(entry)
        
        # Find Test Configuration
        elif vals.find('Test Configuration') == 0:
            value = vals[19::]
            entry = ('TestConfiguration', value)
            my_list.append(entry)
        
        # Find dimensions
        elif vals.find('X-Section') == 0:
            # This only if section is rectangular
            values = find_float(vals)
            entry1 = ('Width', values[0])
            entry2 = ('Depth', values[1])
            my_list.append(entry1)
            my_list.append(entry2)
        
        elif vals.find('Diameter: ') == 0:
            # This only for spiral
            value = find_float(vals)
            my_list.append(('Diameter', value[0]))
        
        elif vals.find('Length: ') == 0:
            values = find_float(vals)
            my_list.append(('L_Inflection', values[0]))
            my_list.append(('L_Measured', values[1]))
        
        elif vals.find('Concrete') == 0:
            value = find_float(vals)
            my_list.append(('fpc', value[0]))
        
        elif vals.find('Transverse Steel:') == 0:
            values = find_float(vals)
            if len(values) == 1:
                # Only value is Yield Strength
                my_list.append(('fyt', values[0]))
                my_list.append(('fsut', 0))
                
            elif len(values) == 2:
                # Missing value is the Grade
                my_list.append(('fyt', values[0]))
                my_list.append(('fsut', values[1]))
                if values[0] > values[1]:
                    print('Check this case, probably fsut = 0')
                
            elif len(values) == 3:
                my_list.append(('fyt', values[1]))
                my_list.append(('fsut', values[2]))
                if values[1] > values[2]:
                    print('Check this case, probably fsut = 0')
                
            else:
                print('Error, values are missing for trans. reinf. Filling with -99')
                my_list.append(('fyl', -99))
                my_list.append(('fsul', -99))
                
        elif vals.find('Longitudinal Steel:') == 0:
            values = find_float(vals)
            if len(values) == 1:
                # Only value is Yield Strength
                my_list.append(('fyl', values[0]))
                my_list.append(('fsul', 0))
                
            elif len(values) == 2:
                # No corner values, no grade
                my_list.append(('fyl', values[0]))
                my_list.append(('fsul', values[1]))
                
            elif len(values) == 3:
                # No corner values, with grade
                my_list.append(('fyt', values[1]))
                my_list.append(('fsut', values[2]))
                if values[1] > values[2]:
                    print('Check this case, probably fsul = 0')
                    
            elif len(values) == 4:
                # Corner values, missing grade
                my_list.append(('fyl', values[0]))
                my_list.append(('fsul', values[1]))
                if values[0] > values[1]:
                    print('Check this case, probably fsul = 0')
                    
            elif len(values) == 5:
                # Corner values, with grade
                my_list.append(('fyl', values[1]))
                my_list.append(('fsul', values[2]))
                if values[1] > values[2]:
                    print('Check this case, probably fsul = 0')
            else:
                print('Error... values are missing for long. reinf. Filling with -99')
                my_list.append(('fyl', -99))
                my_list.append(('fsul', -99))
                
        elif vals.find('Axial Load:') == 0:
            
            # Get the value of the applied axial load during the test
            value = find_float(vals)
            if len(value) > 0:
                my_list.append(('AxLoad', value[0]))
            else:
                my_list.append(('AxLoad', 0.0))
        
        elif vals.find('P-D:') == 0:
            my_list.append(('P_Delta', vals[4::]))
        
        # Properties of the longitudinal reinforcement
        elif vals.find('Diameter:') == 0:
            values = find_float(vals)
            if len(values) <= 1:
                my_list.append(('dlb', values[0]))
            else:
                my_list.append(('dlb', values[0]))
                my_list.append(('dlb_c', values[1]))
        
        elif vals.find('Number of Bars') == 0:
            value = find_float(vals)
            my_list.append(('nlb', value[0]))
        
        # Properties of the transverse reinforcement
        elif vals.find('Number of Shear Legs') == 0:
            values = find_float(vals)
            if len(values) == 1:
                my_list.append(('nsl', values[0]))
            else:
                print('Check number of shear legs, theres more than one value')
        
        elif vals.find('Region of Close') == 0:
            values = find_float(vals)
            if len(values) != 3:
                print('Warning, not enough values found')
            
            my_list.append(('dtb_rcs', values[0]))
            my_list.append(('s_rcs', values[2]))
        
        elif vals.find('Region of Wide') == 0:
            values = find_float(vals)
            if len(values) != 3:
                print('Warning, not enough values found')
            
            my_list.append(('dtb_rws', values[0]))
            my_list.append(('s_rws', values[2]))
            
        elif vals.find('Perpendicular to Load') == 0:
            values = find_float(vals)
            my_list.append(('cc_per', values[0]))
            my_list.append(('nib_per', values[1]))
        
        elif vals.find('Parallel to Load') == 0:
            values = find_float(vals)
            my_list.append(('cc_par', values[0]))
            my_list.append(('nib_par', values[1]))
        
        elif vals.find('Diameter Spiral') == 0:
            value = find_float(vals)
            my_list.append(('dsp', value[0]))
        
        elif vals.find('Hoop Spacing') == 0:
            value = find_float(vals)
            my_list.append(('s', value[0]))
        
        elif vals.find('Cover to Center of Hoop Bar') == 0:
            value = find_float(vals)
            my_list.append(('cc', value[0]))
        
    # Get force-deformation data
    try:
        fd = get_force_deformation(data_url)
        my_list.append(fd)
    except:
        print('Force-Deformation Data not found')
        fd = {
            'disp': [],
            'force': []}
        my_list.append(fd)
        
    return dict(my_list)


# Run the script
if __name__ == '__main__':
    
    current_workdir = os.getcwd()
    files_dir = os.path.join(current_workdir, 'test_data') # r'/Users/miguelgomez/Documents/GitHub/RC_Column_Model/test_data'

    url = 'https://nisee.berkeley.edu/spd/servlet/display?format=html&id='
    
    for ii in range(1, 417):
        
        # (1) Create name of the file
        filename = files_dir + '/test_' + str(ii).zfill(3) + '.json'
        
        # (2) URL to find the information from the test
        url_ii = url + str(ii)
        
        # (3) Get information without format and url of test data
        text_ii, hlk_ii = get_table(url_ii)
        
        # (4) Format test data and put it in a dictionary
        test_data_ii = get_test_data(text_ii, hlk_ii)
        
        # Write JSON object to file for each test
        with open(filename, "w") as outfile: 
            json.dump(test_data_ii, outfile, indent=4)
            print('File created', filename)
        
        
    
    
    
    
    
        












