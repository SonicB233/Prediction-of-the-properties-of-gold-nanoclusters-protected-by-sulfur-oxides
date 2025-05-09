from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):

    indices = list(range(len(dataset)))

#___________________________________________________________________________________
    train_indices = indices[:58]

    val_indices = indices[58:-4]

    test_indices = indices[-4:]
#___________________________________________________________________________________
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader



def collate_pool(dataset_list):

    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids



class GaussianDistance(object):
    
    def __init__(self, dmin, dmax, step, var=None):
        """
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]



class AtomCustomJSONInitializer(AtomInitializer):
  
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):


    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)


    def __len__(self):
        return len(self.id_prop_data)


    @functools.lru_cache(maxsize=None)  # Cache loaded structures

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id+'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea ,bond_angles, dihedral_angles= [], [], [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))


        for i in range(len(crystal)):
            neighbors = all_nbrs[i]

            if len(neighbors) >= 2:
         
                positions = [neighbor[0].coords for neighbor in neighbors]

            
                sorted_j = np.argsort(np.linalg.norm(positions - positions[0], axis=1))
                relpos1 = positions[sorted_j[1]] - positions[sorted_j[0]]
                relpos2 = positions[sorted_j[2]] - positions[sorted_j[0]]

                # 计算相对位置之间的角度的余弦
                cos = np.sum(relpos1 * relpos2) / (np.linalg.norm(relpos1) * np.linalg.norm(relpos2))

                # 修剪余弦值以确保其在有效范围内
                cos = np.clip(cos, -1.0, 1.0)

                # 计算角度并将其转换为度数
                angle = (np.arccos(cos) / np.pi) * 180.0

                bond_angles.append(angle)
            else:
                #如果一个原子的邻居少于两个，则补0
                bond_angles.append(0.0) 




        for i in range(len(crystal)):
            neighbors = all_nbrs[i]

            if len(neighbors) >= 3:
                positions = [neighbor[0].coords for neighbor in neighbors]

                if len(positions) >= 4:
                    p1, p2, p3, p4 = positions[0], positions[1], positions[2], positions[3]
                    b1 = p2 - p1
                    b2 = p3 - p2
                    b3 = p4 - p3
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    m1 = np.cross(n1, b2)
                    x = np.dot(n1, n2)
                    y = np.dot(m1, n2)
                    dihedral_angle = np.arctan2(y, x)
                    dihedral_angles.append(dihedral_angle)
                else:
                    dihedral_angles.append(0.0)  
            else:
                dihedral_angles.append(0.0)  

        # 键角
        bond_angles = np.array(bond_angles)
        bond_angles = torch.Tensor(bond_angles)

        # 二面角
        dihedral_angles = np.array(dihedral_angles)
        dihedral_angles = torch.Tensor(dihedral_angles)

        #原子特征、邻居原子特征、邻居原子索引、target
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])


        last_column_1 = atom_fea[:, -1].unsqueeze(1)
        last_column_1 = torch.cat((last_column_1, dihedral_angles.unsqueeze(1)), 1)
        atom_fea = torch.cat((atom_fea[:, :-1], last_column_1), 1)


        last_column_2 = atom_fea[:, -1].unsqueeze(1)
        last_column_2 = torch.cat((last_column_2, bond_angles.unsqueeze(1)), 1)
        atom_fea = torch.cat((atom_fea[:, :-1], last_column_2), 1)

        #返回值为{[(原子特征、键角、二面角)三者组成的部分，邻居特征，邻居索引]三个部分组成的元组，target，cif_id}
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


