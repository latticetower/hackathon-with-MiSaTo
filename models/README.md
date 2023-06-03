I'll probably need to precompute a vector of distances to ligands before doing anything else. For now I can skip this part.


# Ideas
1. Predict adaptability in the next frame from the embeddings produced by ESM-IF1 previously.

- `colab_notebooks/colab_pyg_dgcnn_v1.ipnynb` does that

As a result of training, we get these metrics (on test part of the dataset):

|              Metric              |             Value on dataset             |
|--------------------------------------|--------------------------------------|
|  Pearson Correlation Coefficient |          0.4146082401275635          |
|               MSE               |         0.19715961813926697          |


2. Predict adaptability mean and std values for each backbone atom  from the embeddings produced by ESM-IF at previous step.

- `colab_notebooks/colab_pyg_dgcnn_v2.ipnynb` will contain the code for training in this setting

3. One of the (1) or (2), but with additional features - distances to top-3 ligand atoms from each atom.
