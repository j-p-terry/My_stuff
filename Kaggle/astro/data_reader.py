import pandas as pd
import numpy as np

"""Reads in data"""
"""Trying to predict target"""

directory_path = "/Users/jasonterry/Documents/Scripts/Misc/My_stuff/" \
                 "Kaggle/astro"

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

def merge(data, metadata):

    """puts data and metadata into one frame"""

    """Wrong, try to put all data for each object on one row
        Take median flux of each passband"""

    # for column in list(metadata):
    #     if column != "object_id":
    #         this_column = np.array([])
    #         for index, row in metadata.iterrows():
    #             for index1, row1 in data.iterrows():
    #                 if row["object_id"] == row1["object_id"]:
    #                     this_column = np.append(this_column, row[column])
    #         data[column] = column

    """Maybe better"""

    new_cols = {"mjd": {}, "flux": {}, "flux_err": {}, "detected": {}}

    for index, row in metadata.iterrows():
        object = row["object_id"]
        # bands = {}
        mjd = {}
        flux = {}
        flux_err = {}
        detected = {}
        for index1, row1 in data.iterrows():
            if row1["object_id"] == object:
                band = row1["passbands"]
                if band in list(mjd.keys()):
                    info = [row1["mjd"], row1["flux"], row1["flux_err"],
                            row1["detected"]]
                    mjd[band] = np.vstack([mjd[band], info][0])
                    flux[band] = np.vstack([flux[band], info[1]])
                    flux_err[band] = np.vstack([flux_err[band], info[2]])
                    detected[band] = np.vstack([detected[band], info[3]])
                else:
                    info = [row1["mjd"], row1["flux"], row1["flux_err"],
                            row1["detected"]]
                    mjd[band] = np.array([info[0]])
                    flux[band] = np.array([info[1]])
                    flux_err[band] = np.array([info[2]])
                    detected[band] = np.array([info[3]])

        object_data = {"mjd": {}, "flux": {}, "flux_err": {}, "detected": {}}
        for band in mjd:
            object_data["mjd"][band] = np.median(mjd[band])
            object_data["flux"][band] = np.median(flux[band])
            object_data["flu_err"][band] = np.median(flux_err[band])
            object_data["detected"][band] = np.mode(detected[band])
            for key in new_cols:
                if band not in new_cols[key]:
                    new_cols[key][band] = np.array(object_data[key][band])
                else:
                    new_cols[key][band] = np.append(new_cols[key][band],
                                                    object_data[key][band])

    for col in new_cols:
        for band in new_cols[col]:
            metadata[col + str(band)] = new_cols[col][band]


    return data

def get_data():

    """Gets and merges data"""

    data, metadata=read(test=False)
    data = merge(data, metadata)
    return data
