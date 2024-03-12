import os, sys
import csv
import numpy as np

from typing import List
from pathlib import Path

class SheetSaver():
    """
    Class for saving logged training and inference data to a file for plotting. 
    One cvs sheet is generated per instance of this class.
    """

    def __init__(self, base_path, sheetname: str, cur_n_anomaly: int, generate_first_column: bool = True, transpose: bool = True, override: bool = True):
        """
        sheetname: Name of the sheet to save to
        cur_n_anomaly: Number of anomalies in the current experiment
        generate_first_column: If True, generate a column of epoch numbers
        transpose: If True, transpose data to fit headers
        override: If True, override existing file
        """

        # Initialize data to save
        self.datasets_to_save: List[np.ndarray] = [] # List of data to save
        self.dataheaders: List[str] = [] # List of headers specified for each dataset

        # Establish path and names
        self.base_path = Path(base_path)
        self.sheetname = Path(sheetname) if sheetname != "" else Path(f"Metrics_{cur_n_anomaly}_anomalies.csv")

        # Establish basic data
        self.cur_n_anomaly = cur_n_anomaly
        self.metrics_path = self.base_path / self.sheetname
        self.generate_first_column = generate_first_column
        self.transpose = transpose
        self.override = override

        # Internal variables
        self.is_first_save = True
        self.is_first_write_after_save = True


    def writeHeaders(self):
        """
        Write headers to new file
        """
        with open(self.metrics_path, mode='w') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(self.dataheaders)


    def addData(self, data, dataheader: str = "No Header"):
        """
        Append new Array to overall List of Data to save,
        and add header to list of headers.
        """
        # If this is the first write after a save, reset the dataheaders and datasets_to_save
        if self.is_first_write_after_save:
            self.dataheaders = []
            self.datasets_to_save = []
            self.is_first_write_after_save = False

        data = np.array(data)
        self.datasets_to_save.append(data)
        self.dataheaders.append(dataheader)

    
    def transposeData(self):
        """
        Transpose data to fit headers
        """
        self.datasets_to_save = list(map(list, zip(*self.datasets_to_save)))


    def save(self):
        if self.is_first_save:
            self.is_first_save = False
            
        # Check if override is True
        if self.metrics_path.exists():
            if not self.override:
                print(f"File {self.metrics_path} already exists, skipping save... (override set to False)")
                return
            else:
                pass
        
        # if generate_first_column is True, generate a column of epoch numbers
        if self.generate_first_column:
            self.dataheaders.insert(0, "Epoch")
            self.datasets_to_save.insert(0, np.arange(len(self.datasets_to_save[0])) + 1) # Add epoch numbers to first column

        # Write headers to file
        self.writeHeaders()

        # Write data to file
        with open(self.metrics_path, mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            if self.transpose: self.transposeData()

            # Write each row to the CSV file
            for row in self.datasets_to_save:
                writer.writerow(row)

        # Reset is_first_write_after_save
        self.is_first_write_after_save = True


        
            


    
