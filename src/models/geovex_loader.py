import h3
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class GeoVexLoader(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        neighbor_k_ring=6,
    ):
        # todo: find all h3's that have all of their neighbors in the dataset,
        # OR, how do we handle h3's that don't have all of their neighbors in the dataset?
        self._valid_h3 = []
        all_indices = set(data.index)

        # number of columns in the dataset
        self._N = data.shape[1]
        self._k = neighbor_k_ring

        for i, (h3_index, hex_data) in tqdm(
            enumerate(data.iterrows()), total=len(data)
        ):
            hex_neighbors_h3 = h3.grid_disk(h3_index, neighbor_k_ring)
            # remove the h3_index from the neighbors
            hex_neighbors_h3.remove(h3_index)
            available_neighbors_h3 = list(hex_neighbors_h3.intersection(all_indices))
            if len(available_neighbors_h3) < len(hex_neighbors_h3):
                # skip adding this h3 as a valid input
                continue
            anchor = np.array(h3.cell_to_local_ij(h3_index, h3_index))
            self._valid_h3.append(
                (
                    h3_index,
                    data.index.get_loc(h3_index),
                    [
                        # get the index of the h3 in the dataset
                        (
                            data.index.get_loc(_h),
                            tuple(
                                (
                                    np.array(h3.cell_to_local_ij(h3_index, _h)) - anchor
                                ).tolist()
                            ),
                        )
                        for _h in hex_neighbors_h3
                    ],
                )
            )

        self._data = data.to_numpy(dtype=np.float32)
        self._data_torch = torch.Tensor(self._data)

    def __len__(self):
        return len(self._valid_h3)

    def __getitem__(self, index):
        # construct the 3d tensor
        h3_index, target_idx, neighbors_idxs = self._valid_h3[index]
        return self._build_tensor(target_idx, neighbors_idxs)

    def _build_tensor(self, target_idx, neighbors_idxs):
        # build the 3d tensor
        # it is a tensor with diagonals of length neighbor_k_ring
        # the diagonals are the neighbors of the target h3
        # the target h3 is in the center of the tensor
        # the tensor is 2*neighbor_k_ring + 1 x 2*neighbor_k_ring + 1 x 2*neighbor_k_ring + 1
        # make a tensor of zeros
        tensor = torch.zeros(
            (
                self._N,
                2 * self._k + 1,
                2 * self._k + 1,
            )
        )

        # set the target h3 to the center of the tensor
        tensor[
            :,
            self._k,
            self._k,
        ] = self._data_torch[target_idx]

        # set the neighbors of the target h3 to the diagonals of the tensor
        for neighbor_idx, (i, j) in neighbors_idxs:
            tensor[:, self._k + i, self._k + j] = self._data_torch[neighbor_idx]

        # return the tensor and the target (which is same as the tensor)
        # should we return the target as a copy of the tensor?
        return tensor


    def full_dataset(self):
        h3s = []
        tensors = []
        for h3_, target_idx, neighbors_idxs in tqdm(self._valid_h3, total=len(self._valid_h3)):
            h3s.append(h3_)
            tensors.append(self._build_tensor(target_idx, neighbors_idxs))
        
        return h3s, torch.stack(tensors)

    @property
    def shape(
        self,
    ) -> int:
        return self._N, (2 * self._k + 1), (2 * self._k + 1)
