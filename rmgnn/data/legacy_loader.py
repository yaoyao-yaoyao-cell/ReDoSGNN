import networkx as nx
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from collections import defaultdict
import os
import numpy as np
import copy
from collections import defaultdict
class GenData(object):
    def __init__(self, g_list, node_labels, graph_labels):
        self.g_list = g_list
        self.node_labels = node_labels
        self.graph_labels = graph_labels
class FileLoader(object):
    def __init__(self, args):
        self.args = args
    def perturb_node_attributes(self, node_attrs, noise_type='gaussian', level=0.1, seed=None):
        """
        :param node_attrs: Raw node attribute rows.
        :param noise_type: Either ``gaussian`` or ``mask``.
        :param level: Perturbation strength, such as 0.1 for ten percent.
        :param seed: Random seed.
        :return: Perturbed node attribute rows.
        """
        if seed is not None:
            np.random.seed(seed)

        perturbed = copy.deepcopy(node_attrs)

        if noise_type == 'gaussian':
            std = np.std(perturbed)
            noise = np.random.normal(0, std * level, size=np.array(perturbed).shape)
            perturbed = (np.array(perturbed) + noise).tolist()

        elif noise_type == 'mask':
            perturbed = np.array(perturbed)
            num_nodes, num_dims = perturbed.shape
            num_nodes_to_mask = int(num_nodes * level)
            masked_nodes = np.random.choice(num_nodes, num_nodes_to_mask, replace=False)

            for node_idx in masked_nodes:
                num_dims_to_mask = int(num_dims * level)
                mask_dims = np.random.choice(num_dims, num_dims_to_mask, replace=False)
                perturbed[node_idx, mask_dims] = 0.0

            perturbed = perturbed.tolist()

        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        return perturbed
    def load_data(self):
        data = self.args.data
        with open('dataset/%s/A.txt' % (data), 'r') as f:
            edges = f.read().splitlines()

        edges = [tuple(map(int, e.replace(" ", "").split(","))) for e in edges]
        print("edges", len(edges))

        with open('dataset/%s/graph_indicator.txt' % (data), 'r') as f:
            g = f.readlines()
        g = [int(i) for i in g]
        print("g", len(g))

        weights = []
        if self.args.edge_weight:
            with open('dataset/%s/edge_labels.txt' % (data), 'r') as f:
                w = f.readlines()
            weights = [int(i) for i in w]
            print("weights:",len(weights))

        with open('dataset/%s/graph_labels.txt' % (data), 'r') as f:
            l = f.readlines()
        graph_labels = [int(i) for i in l]
        print("labels:", len(graph_labels))

        node_labels_path = f'dataset/{data}/node_labels.txt'
        if os.path.exists(node_labels_path):
            with open(node_labels_path, 'r') as f:
                nl = f.readlines()
            node_labels = [int(i) for i in nl]
            print("Loaded node labels:", len(node_labels))
        else:
            print("No node labels found. Using default label 0 for all nodes.")
            node_labels = [0 for _ in range(len(g))]  # len(g) is the node count.
    # Load node attributes.
        node_attrs = []
        attr_path = f'dataset/{data}/node_attributes.txt'
        if os.path.exists(attr_path):
            with open(attr_path, 'r') as f:
                node_attrs = [list(map(float, line.strip().split(','))) for line in f]
            print("Loaded node attributes:", len(node_attrs))
        else:
            print("No node attributes found, proceeding without them.")
        if self.args.perturb and len(node_attrs) > 0:
            noise_type = self.args.perturb_type
            level = self.args.perturb_level
            seed = self.args.seed

            print(f"Applying {noise_type} perturbation with level {level} and seed {seed}")
            node_attrs = self.perturb_node_attributes(
                node_attrs, noise_type=noise_type, level=level, seed=seed
            )
        # Build edges and their weights.
        G_edges, G_weight = [], []
        if self.args.edge_weight:
            for i in tqdm(range(len(graph_labels)), desc="Create edges", unit='graphs'):
                edge = []
                for e in range(len(edges)):
                    if g[edges[e][0] - 1] == i + 1:
                        edge.append(edges[e])
                    elif g[edges[e][0] - 1] == i + 2:
                        break
                G_edges.append(edge)
            for i in tqdm(range(len(graph_labels)), desc="Create weights", unit='graphs'):
                weight = []
                for w in range(len(weights)):
                    if g[edges[w][0]-1] == i + 1:
                        weight.append(weights[w])
                    elif g[edges[w][0]-1] == i + 2:
                        break
                G_weight.append(weight)
        else:
            for i in tqdm(range(len(graph_labels)), desc="Create edges", unit='graphs'):
                edge = []
                weight = []
                for e in range(len(edges)):
                    if g[edges[e][0] - 1] == i + 1:
                        edge.append(edges[e])
                        weight.append(1)
                    elif g[edges[e][0] - 1] == i + 2:
                        break
                G_edges.append(edge)
                G_weight.append(weight)

        # Build the node list belonging to each graph.
        graph_id_to_nodes = defaultdict(list)
        for node_id, graph_id in enumerate(g):
            graph_id_to_nodes[graph_id - 1].append(node_id)  # Graph IDs are one-based.

        g_list = []
        for i in tqdm(range(len(G_edges)), desc="Create original graph", unit='graphs'):
            node_ids = graph_id_to_nodes[i]
            g_list.append(self.gen_graph(G_edges[i], G_weight[i], node_attrs=node_attrs, node_ids=node_ids))

        return GenData(g_list, node_labels, graph_labels)        
    def gen_graph(self, data, weights, node_attrs=None, node_ids=None):
        g1 = [(*edge, w) for edge, w in zip(data, weights)]
        g = nx.Graph()
        g.add_weighted_edges_from(g1)

        if node_attrs is not None and node_ids is not None:
            for i, node in enumerate(sorted(g.nodes())):
                if i >= len(node_ids):
                    # print(f"Warning: node index {i} exceeds node_ids length {len(node_ids)}. Skipping attribute.")
                    continue
                if node_ids[i] >= len(node_attrs):
                    # print(f"Warning: node_ids[{i}] = {node_ids[i]} exceeds node_attrs length {len(node_attrs)}. Skipping attribute.")
                    continue
                g.nodes[node]['attr'] = node_attrs[node_ids[i]]

        return g

class GINDataLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=None,
                 seed=0,):

        self.seed = seed
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}
        print(len(dataset))
        labels = [l for _, l in dataset]
        idx = []
        for i in range(len(labels)):
            idx.append(i)

        sampler = SubsetRandomSampler(idx)

        self.train_loader = GraphDataLoader(
            dataset, sampler=sampler,
            batch_size=batch_size, collate_fn=collate_fn, **self.kwargs)

    def train_valid_loader(self):
        return self.train_loader
