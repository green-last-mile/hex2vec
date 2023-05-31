import h3
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class H3NeighborDataset(Dataset):

    def __init__(self, data: pd.DataFrame, neighbor_k_ring=1, dead_k_ring=2):
        self.data = data
        self.data_torch = torch.Tensor(self.data.to_numpy(dtype=np.float32))
        all_indices = set(data.index)

        self.inputs = []
        self.contexts = []
        self.input_h3 = []
        self.context_h3 = []

        self.positive_indexes = {}

        for i, (h3_index, hex_data) in tqdm(
            enumerate(self.data.iterrows()), total=len(self.data)
        ):
            hex_neighbors_h3 = h3.grid_disk(h3_index, neighbor_k_ring)
            hex_neighbors_h3.remove(h3_index)
            available_neighbors_h3 = list(hex_neighbors_h3.intersection(all_indices))

            contexts_indexes = [
                self.data.index.get_loc(idx) for idx in available_neighbors_h3
            ]

            negative_excluded_h3 = h3.grid_disk(h3_index, dead_k_ring)
            negative_excluded_h3 = list(negative_excluded_h3.intersection(all_indices))
            positive_indexes = [
                self.data.index.get_loc(idx) for idx in negative_excluded_h3
            ]

            self.inputs.extend([i] * len(contexts_indexes))
            self.contexts.extend(contexts_indexes)
            self.positive_indexes[h3_index] = set(positive_indexes)

            self.input_h3.extend([h3_index] * len(available_neighbors_h3))
            self.context_h3.extend(available_neighbors_h3)

        self.inputs = np.array(self.inputs)
        self.contexts = np.array(self.contexts)

        self.input_h3 = np.array(self.input_h3)
        self.context_h3 = np.array(self.context_h3)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.data_torch[self.inputs[index]]
        context = self.data_torch[self.contexts[index]]
        input_h3 = self.input_h3[index]
        neg_index = self.get_random_negative_index(input_h3)
        negative = self.data_torch[neg_index]
        y_pos = 1.0
        y_neg = 0.0

        context_h3 = self.context_h3[index]
        negative_h3 = self.data.index[neg_index]
        return input, context, negative, y_pos, y_neg, input_h3, context_h3, negative_h3

    def get_random_negative_index(self, input_h3):
        excluded_indexes = self.positive_indexes[input_h3]
        negative = np.random.randint(0, len(self.data))
        while negative in excluded_indexes:
            negative = np.random.randint(0, len(self.data))
        return negative

    @property
    def shape(self, ) -> int:
        return self.data_torch.shape



class H3ClusterNeighbor(H3NeighborDataset):

    def __init__(self, data: pd.DataFrame, cluster_labels: pd.Series, ):
        self.data = data
        self.data_torch = torch.Tensor(self.data.to_numpy(dtype=np.float32))
        self.cluster_labels = cluster_labels
        self.h3s = self.data.index.copy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_ = self.data_torch[index]
        label_ = self.cluster_labels[index]

        context_index = self._get_random(label_, pos=True)
        context = self.data_torch[context_index]
        y_pos = 1.0

        negative_index = self._get_random(label_, pos=False)
        negative = self.data_torch[negative_index]
        y_neg = 0.0

        return input_, context, negative, y_pos, y_neg, self.h3s[index], self.h3s[context_index], self.h3s[negative_index]


    def _get_random(self, label, pos=True):
        positive_indexes = self.cluster_labels[(self.cluster_labels == label) if pos else (self.cluster_labels != label)].index
        positive_index = np.random.choice(positive_indexes)
        return positive_index

    



        

    


class H3NeighborDatasetCity(H3NeighborDataset):
    def __init__(self, data: pd.DataFrame, neighbor_k_ring=1, dead_k_ring=2):
    
        self.data = data

        if 'city' not in self.data.columns:
            raise ValueError('Column `city` not found in dataframe. You have to add it before using this dataset, as it groups by city')


        self.data_torch = torch.Tensor(self.data.drop(columns=['city']).to_numpy(dtype=np.float32))

        self.inputs = []
        self.contexts = []
        self.input_h3 = []
        self.context_h3 = []

        self.positive_indexes = {}

        for g_idx, g in tqdm(self.data.groupby('city'), total=len(self.data['city'].unique())):

            all_indices = set(g.index)

            for i, (h3_index, hex_data) in tqdm(
                enumerate(g.iterrows()), total=len(g)
            ):
                
                #  creaet a list of all the neighbors of the current hexagon
                hex_neighbors_h3 = h3.grid_disk(h3_index, neighbor_k_ring)
                hex_neighbors_h3.remove(h3_index)

                #  remove all the neighbors that are not in the current city
                available_neighbors_h3 = list(hex_neighbors_h3.intersection(all_indices))

                
                contexts_indexes = [
                    self.data.index.get_loc(idx) for idx in available_neighbors_h3
                ]

                negative_excluded_h3 = h3.grid_disk(h3_index, dead_k_ring)
                negative_excluded_h3 = list(negative_excluded_h3.intersection(all_indices))
                positive_indexes = [
                    self.data.index.get_loc(idx) for idx in negative_excluded_h3
                ]

                self.inputs.extend([i] * len(contexts_indexes))
                self.contexts.extend(contexts_indexes)
                self.positive_indexes[h3_index] = set(positive_indexes)

                self.input_h3.extend([h3_index] * len(available_neighbors_h3))
                self.context_h3.extend(available_neighbors_h3)

        self.inputs = np.array(self.inputs)
        self.contexts = np.array(self.contexts)

        self.input_h3 = np.array(self.input_h3)
        self.context_h3 = np.array(self.context_h3)





# create dataset loader for the distance prediction task
class H3DistanceDataset(Dataset):

    def __init__(self, data: pd.DataFrame, max_distance: int = 10) -> None:
        
        self.data = data
        self.data_torch = torch.Tensor(self.data.to_numpy(dtype=np.float32))

        self.index_list = list(self.data.index)
        
        self.indice_map = {
            h3_index: i for i, h3_index in enumerate(self.data.index)
        }

        # self.neighbor_map = {
        #     h3_index: h3.grid_disk(h3_index, max_distance).intersection(self.data.index) for h3_index in self.data.index
        # }

        self._k = max_distance

    @property
    def shape(self) -> int:
        return self.data_torch.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        h3_target = self.index_list[index]
        # have a 10% change of getting a neighbor as the target
        if np.random.random() < 0.1:
            # h3_d = np.random.choice(list(neighbors))
            # get the 
            neighbors = h3.grid_disk(h3_target, self._k).intersection(self.index_list)
            h3_d = np.random.choice(list(neighbors))

        else:
            # just get a random h3 index
            h3_d = np.random.choice(self.index_list)

        # return the target, the neighbor, and the distance between them
        try:
            distance = min(h3.h3_distance(h3_target, str(h3_d)), 10)
        except h3.H3ValueError:
            distance = 10
        
        return self.data_torch[index], self.data_torch[self.indice_map[h3_d]], distance 

        # get the index of the h3 index

        





            


