import argparse
import h5py
import pickle
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from pathlib import Path


def write_h5_info(outName, struct, prepared_dict):
    """
    Write features to h5 file. In addition to filtered data, I plan to save indices
    """
    with h5py.File(outName, 'a') as oF:
        if isinstance(prepared_dict, dict):
            subgroup = oF.create_group(struct)
            for preprocessing_property in  prepared_dict.keys():
                if preprocessing_property.startswith('atoms_') or preprocessing_property.startswith('molecules_'):
                    subgroup.create_dataset(preprocessing_property, data = prepared_dict[preprocessing_property], compression = "gzip", dtype='i8')
                elif preprocessing_property.startswith('trajectory_') or preprocessing_property.startswith('feature_'):
                    subgroup.create_dataset(preprocessing_property, data = prepared_dict[preprocessing_property], compression = "gzip", dtype='f8')   
                else:
                    subgroup.create_dataset(preprocessing_property, data = prepared_dict[preprocessing_property], compression = "gzip")
        else:
            # let's suppose we have only trajectory data
            oF.create_dataset(struct, data=prepared_dict, compression = "gzip")


def main(args):
    misato_path = Path(args.misato)
    structs = pickle.load(open(misato_path / "src/data/processing/available_structs.pickle", 'rb'))
    print(len(structs))
    savepath = Path(args.datasetOut)
    if savepath.exists():
        savepath.unlink()
    datadir = Path(args.datasetFolder)
    filenames = list(datadir.glob("*.npz"))
    print("Total filenames in the folder:", len(filenames))
    for filename in tqdm(filenames):
        pdbid = filename.stem
        data = np.load(filename)
        write_h5_info(savepath, pdbid, data['trajectory_coordinates_prepared'])
        
        # print()
        #write_h5_info(savepath, pdbid, dict(data))
        # break


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--datasetFolder", required=False, help="Folder with prepared npz files", default='out', type=str)
    # parser.add_argument("-f", "--datasetFolder", required=False, help="Output dataset in hdf5 format. Will be overwritten if it already exists.", default='out', type=str)
    parser.add_argument("-o", "--datasetOut", required=False, help="Output dataset in hdf5 format. Will be overwritten if it already exists.", default='md_test_out.hdf5', type=str)
    parser.add_argument("-m", "--misato", required=True, help="Folder with the misato dataset - this is needed to reuse mappings from Maps subfolder to do atom names/indices conversion", type=str)
    
    parser.add_argument("-b", "--begin", required=False, help="Start index of structures", default=0, type=int)
    parser.add_argument("-e", "--end", required=False, help="End index of structures", default=9999999, type=int)
    args = parser.parse_args()

    main(args)