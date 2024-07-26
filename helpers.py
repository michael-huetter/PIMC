"""
Some helper functions for the main code.
"""
import numpy as np
import csv
import os
import datetime

def _get_unique_filename(base_filename):
    """
    Get unique filename for save_to_csv function
    """

    counter = 1
    filename, file_extension = os.path.splitext(base_filename)
    
    while os.path.exists(base_filename):
        base_filename = f"{filename}_{counter}{file_extension}"
        counter += 1

    return base_filename

def get_filenames(directory):
    
    files_path = []
    files = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            files.append(filename)
            files_path.append(file_path)
    
    return files_path, files

def save_to_csv(data, filename):
    """
    Save resutls to /output/ with a unique filename if multiple T loops are used
    """

    unique_filename = _get_unique_filename("output/"+str(filename))
    with open(unique_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        if isinstance(data, (list, np.ndarray)):  # If data is a list or numpy array
            for row in data:
                writer.writerow([row] if isinstance(row, np.float64) else row)
        else:  # If data is a single value
            writer.writerow([data])

def wOut(message, filename='output.out'):
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mode = 'a' if os.path.exists(filename) else 'w'

    with open(filename, mode) as file:
        file.write(f'{timestamp} - {message}\n')
        
def remove_all_files_in_folder(folder_path):

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
