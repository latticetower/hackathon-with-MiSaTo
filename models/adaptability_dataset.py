import h5py
import os
import torch
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.data import Data




class AdaptabilityDataset(Dataset):
    def __init__(self, raw_dir, pdbid_list=None, target_pretransform=None, 
                 transform=None, pre_transform=None, pre_filter=None):
        self.pdbid_list = pdbid_list
        self.processed_pdbids = []
        self.pdbid2idx = dict()
        self.file_names = []
        self.target_pretransform = target_pretransform
        super().__init__(raw_dir, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [
            os.path.join(self.raw_dir, 'esm_if_out_frame0.hdf5'), 
            os.path.join(self.raw_dir, 'md_adaptabilities.hdf5')
        ]

    @property
    def processed_file_names(self):
        return self.file_names

    def process(self):
        idx = 0
        embedding_file_name, adaptabilities_file_name = self.raw_file_names
        with h5py.File(embedding_file_name) as embeddings_collection, \
             h5py.File(adaptabilities_file_name) as adaptabilities_collection:
            if self.pdbid_list is not None:
                pdbid_list = self.pdbid_list
            else:
                pdbid_list = list(sorted(self.embeddings_collection.keys()))

            for pdbid in pdbid_list:
                if not pdbid in adaptabilities_collection:
                    continue
                
                if not pdbid in embeddings_collection:
                    continue
                embedding = embeddings_collection[pdbid][()]
                adaptabilities = adaptabilities_collection[pdbid][()]
                embedding = torch.from_numpy(embedding).to(torch.float)
                adaptabilities = torch.from_numpy(adaptabilities).to(torch.float)
                if self.target_pretransform is not None:
                    adaptabilities = self.target_pretransform(adaptabilities)
                data = Data(x=embedding, y=adaptabilities)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(data, os.path.join(self.processed_dir, f'{pdbid}.pt'))

                self.pdbid2idx[pdbid] = idx
                self.processed_pdbids.append(pdbid)
                idx += 1
                self.file_names.append(pdbid)

    def len(self):
        return len(self.file_names)

    def get_indices(self, pdbid_list):
        return np.asarray([self.pdbid2idx[pdbid] for pdbid in pdbid_list if pdbid in self.pdbid2idx])

    def get(self, idx):
        pdbid = self.processed_pdbids[idx]
        data = torch.load(os.path.join(self.processed_dir, f'{pdbid}.pt'))
        return data