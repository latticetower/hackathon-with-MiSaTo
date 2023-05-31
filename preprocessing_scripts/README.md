# Scripts and the order in which they were run on data to preprocess

The scripts itself are based on the ones provided in the MiSaTo dataset repo. Some of them use pickled arrays to convert from pytraj and amber (https://amber-md.github.io/pytraj/latest/index.html) format back to pdb format.

For example, let's suppose we have a full-atom protein chain positions, which I want to convert to the format supported by the particular neural network. 

What I do with each chain's atoms:
- group atoms by residue number,
- for each residue, leave only information on N, CA, C atoms in this particular order and drop information about all other atoms,
- collect this to array with the shape `(n_residues, <3 backbone atoms>, <3 coordinates>)`, for each trajectory stack this becomes array with shape `(n_frames, n_residues, <3 backbone atoms>, <3 coordinates>)`,
- use array this as an input to the model.

MiSaTo sample associated with particular PDB ID might contain more than one protein chain and more than one small molecule. Here I don't rely on PDB information for the particular PDB ID, instead of this I split everything to chains and filter data based on the information provided in MiSaTo repository only, because Protein Data Bank files might have been changed, coordinates might have been fixed, etc.

So for each PDB ID in MiSaTo dataset, I get separate arrays with data on protein chains. Also I store a lot of intermediate data for debugging and analysis purposes.

1. The first script I run is currently called `prepare_h5py_for_esm_if`. It is run as follows:

```
python prepare_h5py_for_esm_if.py -i MiSaTo-dataset/data/MD/h5_files/MD.hdf5 -f out_big_5000 -o "md.hdf5" -m MiSaTo-dataset -b 5000 -e 6000
```
In the example above, it requires the path to particular hdf5 file with data (`-i` parameter), saves everything (including intermediate files) to the one folder (`-f` parameter). Note that in the example above I process only a small portion of dataset (`-b` and `-e` parameters). It is also necessary to provide the folder where the misato dataset is cloned, since the script uses pickled data, stored in one of the dataset's subfolders.

Output: inside of the folder there will be folder called `npz` with preprocessed data (each pdb id correctly processed has the corresponding `<pdbid>.npz` file, which contains the array of trajectories in the format described above and some additional information). Also, there will be the folder with the csv files, which can be used for future exploratorial analysis.

2. Since MiSaTo uses h5py files to store the data, I convert the results from the previous step to one or several hdf5 files (for convenience) with `npz_to_h5.py` script.


3. Notebook `aa_sequences_from_data.ipynb` is used to produce 1-letter protein sequences for different protein chains from intermediate csv files.