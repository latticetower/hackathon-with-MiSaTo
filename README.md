# hackathon-with-MiSaTo
various scripts for the MiSaTo dataset (made by me during the event, might be messy) and notes/draft plots for hackathon.bio

## Dataset

https://github.com/sab148/MiSaTo-dataset

During the hackathon I was implementing models to predict adaptability scores from the protein 3d coordinates.

The dataset contains preprocessed information on 100 frames of molecular dynamics for 16+ thousands of protein-ligand complexes (some of the ligands are the small molecules, the others aren't).

I've decided to use one of large pretrained inverse folding models (in particular, [ESM-IF1](https://github.com/facebookresearch/esm#invf)) to obtain the representations for the atoms in the embeddings space, then to train another model to predict adaptability values from these embeddings.

From the MiSaTo dataset authors' definition (defined by corresponding code in there repo), adaptability scores characterize atomic shifts (in terms of coordinates) relative to frame 0 of MD simulation. Before computing them, full-atom structure from frame N is aligned to full-atom structure from frame 0.

Because the ESM-IF1 uses only backbone atoms as the input, and resulting embeddings provide information on residue level, I've decided to build the model which predict adaptability scores only for the protein backbone atoms.

I also compute adaptability scores using only backbone atom information. It means that during the adaptability computation, when the protein in frame N is aligned to frame 0, this alignment with only backbone atoms differs from the alignment of full-atom structures.

The models I've managed to train during the hackathon on precomputed ESM-IF1 embeddings are in the models folder.

There is a related paper which might help to predict adaptabilities for side chain atoms https://www.pnas.org/doi/10.1073/pnas.2216438120, but I haven't looked into it yet.

## Repo structure

- `preprocessing_scripts`: scripts and jupyter notebooks used for data preprocessing. Description and the order in which they were run is described inside the folder's README.

- `exploratorial` - notebooks with plots computed from preprocessed data.

- `models` - code for models to be run on precomputed data (mostly colab notebooks are used at the moment)

The link to preprocessed data might also be published, but I plan to focus on creation of scripts which can be rerun to reproduce the results, than on storing the data (might publish them on Kaggle or Huggingface though).

