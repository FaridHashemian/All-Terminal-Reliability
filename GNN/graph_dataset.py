import torch
import pandas as pd
import tqdm
from torch_geometric.data import Data, Dataset
import os
import numpy as np
import networkx as nx
import ast
import shutil

class GraphDataset(Dataset):

  def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
    """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
    """
    self.root = root
    if os.path.exists(self.root):
        shutil.rmtree(self.root)
        os.makedirs(os.path.join(self.root, 'raw'))
        os.makedirs(os.path.join(self.root, 'processed'))
    else:
        os.makedirs(os.path.join(self.root, 'raw'))
        os.makedirs(os.path.join(self.root, 'processed'))
    
    self.test = test
    self.filename = filename
    shutil.copyfile(self.filename, os.path.join(self.root, 'raw/'+self.filename))
    super(GraphDataset, self).__init__(root, transform, pre_transform)

  @property
  def raw_file_names(self):
    """ If this file exists in raw_dir, the download is not triggered.
          (The download func. is not implemented here)
    """
    return self.filename

  @property
  def processed_file_names(self):
    """ If these files are found in raw_dir, processing is skipped"""
    self.data = pd.read_csv(self.raw_paths[0], index_col=0, header=None, delimiter=";").reset_index()

    if self.test:
        return [f'data_test_{i}.pt' for i in list(self.data.index)]
    else:
        return [f'data_{i}.pt' for i in list(self.data.index)]

  def download(self):
    pass

  def process(self):
    self.data = pd.read_csv(self.raw_paths[0], header=None, delimiter=";")
    for index, graph in tqdm(self.data.iterrows(), total=self.data.shape[0]):
      graph_obj = graph[0]
      # Get node features
      node_feats = self._get_node_features(graph_obj)
      # Get edge features
      edge_feats = self._get_edge_features(graph_obj)
      # Get adjacency info
      edge_index = self._get_adjacency_info(graph_obj)
      # Get labels info
      label = self._get_labels(graph[1])

      # Create data object
      data = Data(x=node_feats,
                  edge_index=edge_index,
                  edge_attr=edge_feats,
                  y=label
                  )
      if self.test:
        torch.save(data,
                os.path.join(self.processed_dir,
                              f'data_test_{index}.pt'))
      else:
        torch.save(data,
                os.path.join(self.processed_dir,
                              f'data_{index}.pt'))

  
  
  def _get_node_features(self, graph):
    def dic_to_graph(dic):
      transform = nx.Graph(ast.literal_eval(dic))
      return transform

    all_node_feats = []

    G = dic_to_graph(graph)
    weights = nx.get_edge_attributes(G, 'reliability')
    #phi = float(max(nx.adjacency_spectrum(G, weight='weight')))
    #node_centerality = nx.katz_centrality(G, 1/phi-0.1, weight='weight', max_iter=1000)
    #node_coeffecient = nx.clustering(G, weight='weight')

    #####
    #laplacian_matrix = nx.laplacian_matrix(G).todense()
    #laplacian_matrix = (-nx.laplacian_matrix(G).todense().min() + nx.laplacian_matrix(G).todense()).reshape(-1)

    for node in G.nodes():
      node_feats = []
      node_feats.append(1)
      '''
      ######
      for i in range(10):
      #list(laplacian_matrix[node]):
        try:
          node_feats.append(list(laplacian_matrix[node])[i])
        except:
          node_feats.append(1)
      ######
      '''
      #node_feats.append(node_centerality[node])
      #node_feats.append(node_coeffecient[node])
        # Append node features to matrix
      all_node_feats.append(node_feats)

    all_node_feats = np.asarray(all_node_feats)

    pre_out_node_feature = torch.tensor(all_node_feats, dtype=torch.float).reshape(-1)
    out_node_feature = pre_out_node_feature.type(torch.LongTensor)

    return out_node_feature

  
  
  def _get_edge_features(self, graph):
    """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
    """
    def dic_to_graph(dic):
      transform = nx.Graph(ast.literal_eval(dic))
      return transform
    G = dic_to_graph(graph)
    weights = nx.get_edge_attributes(G, 'reliability')

    all_edge_feats = []
    for e in G.edges():
      edge_feats = []
      edge_feats.append(weights[e]*100)
      all_edge_feats += [edge_feats, edge_feats]

    all_edge_feats = np.asarray(all_edge_feats)
    pre_out_edge_feature = torch.tensor(all_edge_feats, dtype=torch.float)
    out_edge_feature = pre_out_edge_feature.squeeze()
    return out_edge_feature

  def _get_adjacency_info(self, graph):
    """
      We could also use rdmolops.GetAdjacencyMatrix(mol)
      but we want to be sure that the order of the indices
      matches the order of the edge features
    """
    def dic_to_graph(dic):
      transform = nx.Graph(ast.literal_eval(dic))
      return transform
    G = dic_to_graph(graph)

    edge_indices = []

    for e in G.edges():
      i = e[0]
      j = e[1]
      edge_indices += [[i, j], [j, i]]
    edge_indices = torch.tensor(edge_indices)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    return edge_indices

  def _get_labels(self, label):
    label = np.asarray([label])
    return torch.tensor(label, dtype=torch.float)

  def len(self):
    return self.data.shape[0]

  def get(self, idx):
    """ - Equivalent to __getitem__ in pytorch
          - Is not needed for PyG's InMemoryDataset
    """
    if self.test:
        data = torch.load(os.path.join(self.processed_dir,
                                f'data_test_{idx}.pt'))
    else:
        data = torch.load(os.path.join(self.processed_dir,
                                 f'data_{idx}.pt'))
    return data
