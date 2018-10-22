import pandas as pd
import numpy as np

from .data_reader import read, get_data, merge

"""Gets data from data_reader.py and cleans it"""

directory_path = "/Users/jasonterry/Documents/Scripts/Misc/My_stuff/" \
                 "Kaggle/astro"

metadata_columns = ["object_id", "ra", "decl", "gal_l", "gal_b"	"ddf",
                    "hostgal_specz", "hostgal_photoz", "hostgal_photoz_err",
                    "distmod", "mwebv", "target"]

data_columns = ["object_id", "mjd", "passband", "flux", "flux_err",
                "detected"]


def compare_galaxies(test=False):

    """Looks at entries with zero distance modulus or galaxy spectra
    and impute them with values from objects in same galaxy or drops if
    there are no other objects in that galaxy"""

    data, metadata = read(test)

    galaxy_cols = ["hostgal_specz", "hostgal_photoz_err", "distmod",
                   "gal_l", "gal_b"]

    for index, row in metadata.iterrows():
        if row["hostgal_specz"] == 0 or row["distmod"].isnull():
            for index1, row1 in metadata.iterrows():
                if row["gal_l"] == row1["gal_l"] and \
                 row["gal_b"] == row1["gal_b"]:
                    for feature in galaxy_cols[:3]:
                        row[feature] = row1[feature]

    metadata = metadata.dropna()

    if test:
        set = "test_"
    else:
        set = "train_"

    new_data = merge(data, metadata)
    new_data.to_csv(directory_path + "/data/" + set + "galaxy_merged.csv",
                    index=False)

