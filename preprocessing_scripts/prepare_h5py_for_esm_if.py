"""
Example how to run:

    python prepare_h5py_for_esm_if.py -i MiSaTo-dataset/data/MD/h5_files/MD.hdf5 -f out_big_5000 -o "md.hdf5" -m MiSaTo-dataset -b 5000 -e 6000

in the example above -i relative (or absolute) path to the dataset we want to process;
-f is for the folder where all the results are saved;
-m is the path to the root folder with MiSaTo dataset (can be cloned from github)
-b and -e are start and end indices of the pdb ids found in the input file

Currently script skips the pdb ids which have any missing positions of N, CA, C atoms.
It saves csv files with informations on positions, including the cases with missing data.
The structures where all the backbone atoms positions are present are saved to .npz files in the output folders. 
The stored data mostly includes the trajectory points array, reshaped to <frames, residues in protein chains, [N, CA, C], [x, y, z]>
Also the indices of chains are stored.

"""

import pickle
import h5py
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import os
import sys
from tqdm.auto import tqdm


atomic_numbers_Map = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O',11:'Na',12:'Mg',14:'Si',15:'P',16:'S',17:'Cl',19:'K',20:'Ca',35:'Br',53:'I'}


def get_maps(mapdir):
    residueMap = pickle.load(open(mapdir/'atoms_residue_map.pickle','rb'))
    typeMap = pickle.load(open(mapdir/'atoms_type_map.pickle','rb'))
    nameMap = pickle.load(open(mapdir/'atoms_name_map_for_pdb.pickle','rb'))
    return residueMap, typeMap, nameMap


def update_residue_indices(residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index,residue_Map, typeMap):
    """
    If the atom sequence has O-N icnrease the residueNumber
    """
    if i < len(atoms_type)-1:
        if type_string == 'O' and typeMap[atoms_type[i+1]] == 'N' or residue_Map[atoms_residue[i+1]]=='MOL':
            # GLN has a O N sequence within the AA
            if not ((residue_name == 'GLN' and residue_atom_index==12) or (residue_name == 'ASN' and residue_atom_index==9)):
                residue_number +=1
                residue_atom_index = 0
    return residue_number, residue_atom_index

def insert_TERS(i, molecules_begin_atom_index, residue_number, residue_atom_index, lines):
    """
    We have to insert TERs for the endings of the molecule
    """
    if i+1 in molecules_begin_atom_index:
        lines.append({
            'TER': 1,
            "serial": i+1,
            "atom_name": "TER",
            "residue_name": "TER",
            "residue_number": residue_number,
            "atomic_numbers": 0, # residue_atom_index,
            "residue_atom_index": residue_atom_index
        })
        residue_number +=1
        residue_atom_index = 0
    return residue_number, residue_atom_index, lines


def get_atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, nameMap, verbose=False):
    if residue_name == 'MOL':
        # print("MOL", atoms_number[i], atomic_numbers_Map.get(atoms_number[i]), type_string)
        atom_name = atomic_numbers_Map.get(atoms_number[i], type_string)+str(residue_atom_index)
    else:
        try:
            atom_name = nameMap[(residue_name, residue_atom_index-1, type_string)]
        except KeyError:
            if verbose:
                print('KeyError', (residue_name, residue_atom_index-1, type_string))
                print(atoms_number[i])
            atom_name = atomic_numbers_Map[atoms_number[i]]+str(residue_atom_index)
    return atom_name


def create_df_from_MD(atoms_type, atoms_number, atoms_residue, molecules_begin_atom_index, typeMap,residue_Map, nameMap):
    """
    this is similar to the method create_pdb_lines_MD, except it doesn't save structure to pdb file, instead of 
    
    """
    lines = []
    residue_number = 1
    residue_atom_index = 0
    for i in range(len(atoms_type)):
        residue_atom_index +=1
        type_string = typeMap[atoms_type[i]]
        residue_name = residue_Map[atoms_residue[i]]
        atom_name = get_atom_name(i, atoms_number, residue_atom_index, residue_name, type_string, nameMap)
        # x,y,z = trajectory_coordinates[i][0],trajectory_coordinates[i][1],trajectory_coordinates[i][2]
        # print(atoms_number[i])
        # print(i+1,
        #     atom_name,residue_name,
        #     residue_number,x,y,z,
        #     atomic_numbers_Map[atoms_number[i]])
        # print(atomic_numbers_Map[atoms_number[i]])
        # line = 'ATOM{0:7d}  {1:<4}{2:<4}{3:>5}    {4:8.3f}{5:8.3f}{6:8.3f}  1.00  0.00           {7:<5}'.format(
        #     i+1,
        #     atom_name,residue_name,
        #     residue_number,x,y,z,
        #     atomic_numbers_Map[atoms_number[i]])
        
        record = {
            "serial": i+1,
            "atom_name": atom_name,
            "residue_name": residue_name,
            "residue_number": residue_number,
            "atomic_numbers": atoms_number[i],
            "residue_atom_index": residue_atom_index
        }
        residue_number, residue_atom_index = update_residue_indices(residue_number, i, type_string, atoms_type, atoms_residue, residue_name, residue_atom_index,residue_Map, typeMap)
        lines.append(record)
        residue_number, residue_atom_index, lines = insert_TERS(i, molecules_begin_atom_index, residue_number, residue_atom_index, lines)
    return pd.DataFrame(lines)

def write_h5_info(outName, struct, prepared_dict):
    """
    Write features to h5 file. In addition to filtered data, I plan to save indices
    """
    with h5py.File(outName, 'a') as oF:
        subgroup = oF.create_group(struct)
        for preprocessing_property in  prepared_dict.keys():
            if preprocessing_property.startswith('atoms_') or preprocessing_property.startswith('molecules_'):
                subgroup.create_dataset(preprocessing_property, data = prepared_dict[preprocessing_property], compression = "gzip", dtype='i8')
            elif preprocessing_property.startswith('trajectory_') or preprocessing_property.startswith('feature_'):
                subgroup.create_dataset(preprocessing_property, data = prepared_dict[preprocessing_property], compression = "gzip", dtype='f8')   
            else:
                subgroup.create_dataset(preprocessing_property, data = prepared_dict[preprocessing_property], compression = "gzip")
        # for h5_property in  h5_entries.keys():
        #     if h5_property.startswith('frames_'):
        #         subgroup.create_dataset(h5_property, data= h5_entries[h5_property], compression = "gzip", dtype='f8')


def main(args):
    misato_path = Path(args.misato)
    residueMap, typeMap, nameMap = get_maps(misato_path / "src/data/processing/Maps")
    structs = pickle.load(open(misato_path / "src/data/processing/available_structs.pickle", 'rb'))
    data_collection = h5py.File(args.datasetIn)
    savedir = Path(args.datasetFolder)
    savedir.mkdir(exist_ok=True)
    pdbdir = savedir / "pdb"
    pdbdir.mkdir(exist_ok=True)
    (pdbdir / "collected").mkdir(exist_ok=True)
    savepath = savedir / args.datasetOut
    selected_structs = set(structs[args.begin:args.end]) & set(data_collection.keys())
    print("selected structures:", len(selected_structs))
    selected_structs = sorted(selected_structs)
    skipped_list = []
    npdir = savedir / "npz"
    npdir.mkdir(exist_ok=True)

    for struct in tqdm(selected_structs):
        # if struct in data_collection:
        protein = data_collection[struct]
        atoms_type = protein['atoms_type']
        atoms_number = protein['atoms_number']
        atoms_residue = protein['atoms_residue']
        molecules_begin_atom_index = protein['molecules_begin_atom_index']
        trajectory_coordinates = protein['trajectory_coordinates']

        df = create_df_from_MD(
            atoms_type, atoms_number,
            atoms_residue, molecules_begin_atom_index,
            typeMap, residueMap, nameMap)
        
        df['chain'] = df.TER.fillna(0).cumsum().astype(int)
        df = df.loc[df.TER.isnull()].drop(columns="TER").reset_index(drop=True)
        df.to_csv(pdbdir / f"{struct}.csv", index=None)
        ids = (df.residue_name!= "MOL") & df.atom_name.isin(["N", "CA", "C"])
        aggregated_df = df.reset_index()[ids].pivot(
            index=["chain", 'residue_number', 'residue_name'], 
            columns='atom_name', values='index').reset_index().sort_values(by=["chain", "residue_number"])
        aggregated_df.to_csv(pdbdir / "collected" / f"{struct}.csv", index=None)
        collected_index = aggregated_df[["N", "CA", "C"]].values
        if np.isnan(collected_index).any():
            print(f"Some coordinates are not defined for {struct}, skip")
            skipped_list.append(struct)
            continue

        prepared_trajectory_coordinates = np.take(trajectory_coordinates, collected_index, axis=1)
        residue_numbers = aggregated_df.residue_number.values
        chain_numbers = aggregated_df.chain.values
        residue_names = aggregated_df.residue_name.values
        prepared_dict = {
            'trajectory_coordinates_prepared': prepared_trajectory_coordinates,
            'atoms_residues': residue_numbers,
            'atoms_chain_numbers': chain_numbers,
        }
        savepath = npdir / f"{struct}.npz"
        if savepath.exists() and savepath.is_file():
            savepath.unlink()
        np.savez_compressed(savepath.as_posix(), **prepared_dict)
        
        # write_h5_info(savepath, struct, prepared_dict)

    with open(savedir/"skipped.txt", 'w') as f:
        for skip in skipped_list:
            f.write(skip+"\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--datasetIn", required=False, help="MISATO dataset path to read from in hdf5 format.", default='MD_dataset_mapped.hdf5', type=str)
    parser.add_argument("-f", "--datasetFolder", required=False, help="Output dataset in hdf5 format. Will be overwritten if it already exists.", default='out', type=str)
    parser.add_argument("-o", "--datasetOut", required=False, help="Output dataset in hdf5 format. Will be overwritten if it already exists.", default='MD_dataset_mapped_stripped.hdf5', type=str)
    parser.add_argument("-m", "--misato", required=True, help="Folder with the misato dataset - this is needed to reuse mappings from Maps subfolder to do atom names/indices conversion", type=str)
    
    parser.add_argument("-b", "--begin", required=False, help="Start index of structures", default=0, type=int)
    parser.add_argument("-e", "--end", required=False, help="End index of structures", default=9999999, type=int)
    args = parser.parse_args()

    main(args)