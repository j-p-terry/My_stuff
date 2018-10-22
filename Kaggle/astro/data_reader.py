import pandas as pd
import numpy as np

"""Reads in data"""
"""Trying to predict target"""

directory_path = "/Users/jasonterry/Documents/Scripts/Misc/Kaggle/astro"

metadata_columns = ["object_id", "ra", "decl", "gal_l", "gal_b"	"ddf",
                    "hostgal_specz", "hostgal_photoz", "hostgal_photoz_err",
                    "distmod", "mwebv", "target"]

data_columns = ["object_id", "mjd", "passband", "flux", "flux_err",
                "detected"]


def read(test=False):

    """Reads in data and metadata"""

    if test:
        data = pd.read_csv(directory_path + '/data/test_set_sample.csv')
        metadata = pd.read_csv(directory_path +
                               "/data/test_set_metadata.csv")

    else:
        data = pd.read_csv(directory_path + "/data/training_set.csv")
        metadata = pd.read_csv(directory_path +
                               "/data/training_set_metadata.csv")

    return data, metadata

def merge():

    """puts data and metadata into one frame"""

    data, metadata = read(test=False)

    for column in list(metadata):
        if column != "object_id":
            this_column = np.array([])
            for index, row in metadata.iterrows():
                for index1, row1 in data.iterrows():
                    if row["object_id"] == row1["object_id"]:
                        this_column = np.append(this_column, row[column])
            data[column] = column

    return data


