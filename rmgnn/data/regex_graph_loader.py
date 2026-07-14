"""Load TU-style heterogeneous regex graphs for RMGNN training."""

import networkx as nx
import torch
import os
from tqdm import tqdm
import math
import numpy as np
import copy
import gc
from itertools import product
def s_numeric(x, y, range_val):
    if x is None or y is None:
        return None
    if range_val <= 0:
        return 1.0
    return 1 - abs(x - y) / range_val

def s_categorical(x, y):
    if x is None or y is None:
        return None
    return 1.0 if x == y else 0.0

def s_d_prime(base_sim, gamma):
    return math.exp(-gamma * (1 - base_sim))

def P(attr1, attr2, attr_meta, gamma):
    total, denom = 0.0, 0.0
    for d, meta in attr_meta.items():
        x, y = attr1.get(d), attr2.get(d)
        if meta['type'] == 'numeric':
            base = s_numeric(x, y, meta['range'])
        else:
            base = s_categorical(x, y)
        if base is None:
            continue
        w = meta.get('weight', 1.0)
        total += w * s_d_prime(base, gamma)
        denom += w
    return (total / denom) if denom > 0 else 0.0


def extract_star(G: nx.Graph, center, k=1):
    nodes = nx.single_source_shortest_path_length(G, center, cutoff=k).keys()
    return G.subgraph(nodes).copy(), center

def k_s(S1: nx.Graph, center1, S2: nx.Graph, center2,
        attr_meta_nodes, attr_meta_edges, gamma, eps=1e-8):
    Pc = P(S1.nodes[center1], S2.nodes[center2], attr_meta_nodes, gamma)
    if Pc <= math.exp(-gamma) + eps:
        return 0.0

    elems1 = [('node', S1.nodes[n]) for n in S1.nodes] + [('edge', d) for _, _, d in S1.edges(data=True)]
    elems2 = [('node', S2.nodes[n]) for n in S2.nodes] + [('edge', d) for _, _, d in S2.edges(data=True)]

    total = 0.0
    for (t1, a1), (t2, a2) in product(elems1, elems2):
        if t1 != t2:
            continue
        meta = attr_meta_nodes if t1 == 'node' else attr_meta_edges
        total += P(a1, a2, meta, gamma)
    return total

def K_S(G: nx.Graph, H: nx.Graph, attr_meta_nodes, attr_meta_edges, gamma):
    total = 0.0
    for v in G.nodes:
        S1, _ = extract_star(G, v, k=1)
        for u in H.nodes:
            S2, _ = extract_star(H, u, k=1)
            total += k_s(S1, v, S2, u, attr_meta_nodes, attr_meta_edges, gamma)
    return total

def K_NAS(G: nx.Graph, H: nx.Graph, attr_meta_nodes, attr_meta_edges, gamma, max_hops):
    total = 0.0
    for h in range(1, max_hops + 1):
        for v in G.nodes:
            S1, _ = extract_star(G, v, k=h)
            for u in H.nodes:
                S2, _ = extract_star(H, u, k=h)
                total += k_s(S1, v, S2, u, attr_meta_nodes, attr_meta_edges, gamma)
    return total

def perform_NAS_graph_kernel_computation(G1, G2, gamma, H, attr_meta_nodes, attr_meta_edges):
    return K_NAS(G1.g, G2.g, attr_meta_nodes, attr_meta_edges, gamma=gamma, max_hops=H)



class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.g = g
        self.label = label
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0
        self.max_neighbor = 0


def _load_raw_regex_data(dataset, degree_as_tag, data_dir):
    base = os.path.join(data_dir, dataset)
    required = ('A.txt', 'graph_indicator.txt', 'graph_labels.txt')
    if not all(os.path.exists(os.path.join(base, name)) for name in required):
        return None

    with open(os.path.join(base, 'graph_indicator.txt'), encoding='utf-8') as f:
        indicator_raw = [int(line.strip()) for line in f if line.strip()]
    graph_base = min(indicator_raw)
    indicators = [value - graph_base for value in indicator_raw]

    with open(os.path.join(base, 'graph_labels.txt'), encoding='utf-8') as f:
        graph_labels_raw = [int(line.strip()) for line in f if line.strip()]
    label_map = {value: idx for idx, value in enumerate(sorted(set(graph_labels_raw)))}

    attr_path = os.path.join(base, 'node_attributes.txt')
    if os.path.exists(attr_path):
        with open(attr_path, encoding='utf-8') as f:
            attrs = [
                np.asarray([float(x) for x in line.strip().split(',')], dtype=np.float32)
                for line in f if line.strip()
            ]
    else:
        attrs = [np.zeros(0, dtype=np.float32) for _ in indicators]
    if len(attrs) != len(indicators):
        raise ValueError(f"{attr_path} row count does not match graph_indicator.txt")

    node_label_path = os.path.join(base, 'node_labels.txt')
    if os.path.exists(node_label_path):
        with open(node_label_path, encoding='utf-8') as f:
            raw_node_labels = [int(line.strip()) for line in f if line.strip()]
    else:
        raw_node_labels = [0] * len(indicators)

    with open(os.path.join(base, 'A.txt'), encoding='utf-8') as f:
        raw_edges = [
            tuple(map(int, line.replace(' ', '').strip().split(',')))
            for line in f if line.strip()
        ]
    node_base = min((min(edge) for edge in raw_edges), default=0)
    edges = [(u - node_base, v - node_base) for u, v in raw_edges]

    edge_types = [1] * len(edges)
    edge_label_path = os.path.join(base, 'edge_labels.txt')
    if os.path.exists(edge_label_path):
        with open(edge_label_path, encoding='utf-8') as f:
            loaded = [int(line.strip()) for line in f if line.strip()]
        if len(loaded) == len(edges):
            edge_types = loaded

    graph_nodes = {}
    for global_id, graph_id in enumerate(indicators):
        graph_nodes.setdefault(graph_id, []).append(global_id)

    graphs = []
    global_to_local = {}
    feat_dict = {}
    for graph_id, raw_label in enumerate(graph_labels_raw):
        nx_graph = nx.Graph()
        node_tags = []
        semantic_rows = []
        for local_id, global_id in enumerate(graph_nodes.get(graph_id, ())):
            global_to_local[global_id] = (graph_id, local_id)
            attr = attrs[global_id]
            raw_tag = raw_node_labels[global_id] if global_id < len(raw_node_labels) else 0
            if len(attr) >= 16 and float(attr[:16].sum()) > 0:
                raw_tag = int(np.argmax(attr[:16]))
            if raw_tag not in feat_dict:
                feat_dict[raw_tag] = len(feat_dict)
            node_tags.append(feat_dict[raw_tag])
            semantic_rows.append(attr)
            nx_graph.add_node(
                local_id,
                attr=attr.tolist(),
                node_type=raw_tag,
                global_id=global_id,
            )
        if semantic_rows and any(row.size for row in semantic_rows):
            max_dim = max(row.size for row in semantic_rows)
            padded_rows = []
            for row in semantic_rows:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:row.size] = row
                padded_rows.append(padded)
            semantic = np.stack(padded_rows)
        else:
            semantic = None
        graphs.append(S2VGraph(nx_graph, label_map[raw_label], node_tags, semantic))

    for (global_u, global_v), edge_type in zip(edges, edge_types):
        if global_u not in global_to_local or global_v not in global_to_local:
            raise ValueError(f"Invalid edge ({global_u}, {global_v}) in {dataset}")
        graph_u, local_u = global_to_local[global_u]
        graph_v, local_v = global_to_local[global_v]
        if graph_u != graph_v:
            raise ValueError(f"Cross-graph edge ({global_u}, {global_v}) in {dataset}")
        nx_graph = graphs[graph_u].g
        relation = 'child' if edge_type == 1 else ('ref' if edge_type == 2 else f'rel_{edge_type}')
        if nx_graph.has_edge(local_u, local_v):
            nx_graph.edges[local_u, local_v].setdefault('relations', set()).add(relation)
        else:
            nx_graph.add_edge(
                local_u, local_v, type=edge_type, weight=float(edge_type),
                relations={relation},
            )

    for graph in graphs:
        graph.neighbors = [[] for _ in range(len(graph.g))]
        for u, v in graph.g.edges():
            graph.neighbors[u].append(v)
            graph.neighbors[v].append(u)
        graph.max_neighbor = max((len(x) for x in graph.neighbors), default=0)
        directed_edges = [[u, v] for u, v in graph.g.edges()]
        directed_edges += [[v, u] for u, v in graph.g.edges()]
        graph.edge_mat = (
            torch.LongTensor(directed_edges).t()
            if directed_edges else torch.empty((2, 0), dtype=torch.long)
        )

    if degree_as_tag:
        for graph in graphs:
            graph.node_tags = list(dict(graph.g.degree).values())

    tagset = sorted({tag for graph in graphs for tag in graph.node_tags})
    tag2idx = {tag: idx for idx, tag in enumerate(tagset)}
    for graph in graphs:
        tag_features = torch.zeros(len(graph.node_tags), len(tagset))
        for idx, tag in enumerate(graph.node_tags):
            tag_features[idx, tag2idx[tag]] = 1
        if isinstance(graph.node_features, np.ndarray):
            semantic = torch.as_tensor(graph.node_features, dtype=torch.float32)
            graph.node_features = torch.cat((tag_features, semantic), dim=1)
        else:
            graph.node_features = tag_features

    return graphs, len(label_map)


def load_data(dataset, degree_as_tag,
              attr_meta_nodes, attr_meta_edges,
              data_dir='dataset'):
    """Load regex graphs and return them with the number of label classes."""
    raw_data = _load_raw_regex_data(dataset, degree_as_tag, data_dir)
    if raw_data is not None:
        return raw_data

    g_list = []
    label_dict = {}
    feat_dict = {}
    path = f"{data_dir}/{dataset}/{dataset}.txt"
    with open(path, 'r') as f:
        n_g = int(f.readline().strip())
        for _ in range(n_g):
            n, l = map(int, f.readline().split())
            if l not in label_dict:
                label_dict[l] = len(label_dict)
            g = nx.Graph()
            node_tags, node_features = [], []
            for j in range(n):
                g.add_node(j)
                parts = f.readline().split()
                deg = int(parts[1])
                tmp = deg + 2
                if tmp == len(parts):
                    parts_idx = list(map(int, parts))
                    attr = None
                else:
                    parts_idx = list(map(int, parts[:tmp]))
                    attr = np.array(list(map(float, parts[tmp:])))
                tag = parts_idx[0]
                if attr is not None and len(attr) >= 16 and np.sum(attr[:16]) > 0:
                    tag = int(np.argmax(attr[:16]))
                if tag not in feat_dict:
                    feat_dict[tag] = len(feat_dict)
                node_tags.append(feat_dict[tag])
                if attr is not None:
                    node_features.append(attr)
                for k in parts_idx[2:]:
                    if g.has_edge(j, k):
                        g.edges[j, k].setdefault('relations', set()).add('child')
                    else:
                        g.add_edge(j, k, type=1, weight=1.0, relations={'child'})
            if node_features:
                node_features = np.stack(node_features)
                feature_flag = True

                for node_id in range(n):
                    g.nodes[node_id]['attr'] = node_features[node_id].tolist()
                    if node_features.shape[1] >= 16:
                        g.nodes[node_id]['node_type'] = int(np.argmax(node_features[node_id, :16]))

                # Recover semantic relation labels that the legacy graph-text
                # format cannot store explicitly.
                captures = {}
                for node_id in range(n):
                    node_type = g.nodes[node_id].get('node_type', 0)
                    capture_index = int(round(node_features[node_id, -1]))
                    if node_type in (9, 10) and capture_index >= 0:
                        captures[capture_index] = node_id
                for node_id in range(n):
                    node_type = g.nodes[node_id].get('node_type', 0)
                    target_index = int(round(node_features[node_id, -1]))
                    if node_type == 15 and target_index in captures:
                        target = captures[target_index]
                        if not g.has_edge(node_id, target):
                            g.add_edge(node_id, target)
                        g.edges[node_id, target]['type'] = 2
                        g.edges[node_id, target]['weight'] = 2.0
                        g.edges[node_id, target]['relations'] = {'ref'}

            else:
                node_features = None
                feature_flag = False
            g_list.append(S2VGraph(g, label_dict[l], node_tags, node_features))
    for g in g_list:
        g.neighbors = [[] for _ in range(len(g.g))]
        for i,j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        g.max_neighbor = max((len(nb) for nb in g.neighbors), default=0)
        edges = [ [i,j] for i,j in g.g.edges() ]
        edges += [[j,i] for i,j in g.g.edges()]
        g.edge_mat = torch.LongTensor(edges).t()
    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
    # Preserve the paper's heterogeneous semantic attributes. For legacy
    # datasets without attributes, fall back to one-hot node tags.
    tagset = sorted({tag for g in g_list for tag in g.node_tags})
    tag2idx = {tag:i for i,tag in enumerate(tagset)}
    for g in g_list:
        tag_ft = torch.zeros(len(g.node_tags), len(tagset))
        for i,tag in enumerate(g.node_tags):
            tag_ft[i, tag2idx[tag]] = 1
        if isinstance(g.node_features, np.ndarray):
            semantic_ft = torch.as_tensor(g.node_features, dtype=torch.float32)
            g.node_features = torch.cat((tag_ft, semantic_ft), dim=1)
        else:
            g.node_features = tag_ft

    return g_list, len(label_dict) 

class GenGraph(object):
    def __init__(self, data, num_graphs):
        self.data = data
        self.nodes_labels = data.node_labels
        self.vocab = {}
        self.whole_node_count = {}
        self.weight_vocab = {}
        self.node_count = {}
        self.edge_count = {}
        self.g_final = self.gen_components(num_graphs)
        self.num_cliques = self.g_final.number_of_nodes() - len(self.data.g_list)
        del self.data, self.vocab, self.whole_node_count, self.weight_vocab, self.node_count, self.edge_count
        gc.collect()
    def gen_components(self, num_graphs):
        g_list = self.data.g_list
        h_g = nx.Graph()
        for g in tqdm(range(len(g_list)), desc='Gen Components', unit='graph'):
            clique_list = []
            mcb = nx.cycle_basis(g_list[g])
            mcb_tuple = [tuple(ele) for ele in mcb]
            edges = list(g_list[g].edges())[:num_graphs]  # Only use the first num_graphs edges
            for e in edges:
                weight = g_list[g].get_edge_data(e[0], e[1])['weight']
                edge = ((self.nodes_labels[e[0] - 1], self.nodes_labels[e[1] - 1]), weight)
                clique_id = self.add_to_vocab(edge)
                clique_list.append(clique_id)
                if clique_id not in self.whole_node_count:
                    self.whole_node_count[clique_id] = 1
                else:
                    self.whole_node_count[clique_id] += 1

            for m in mcb_tuple:
                weight = tuple(self.find_ring_weights(m, g_list[g]))
                ring = [self.nodes_labels[m[i] - 1] for i in range(len(m))]
                cycle = (tuple(ring), weight)
                cycle_id = self.add_to_vocab(cycle)
                clique_list.append(cycle_id)
                if cycle_id not in self.whole_node_count:
                    self.whole_node_count[cycle_id] = 1
                else:
                    self.whole_node_count[cycle_id] += 1

            for e in clique_list:
                self.add_weight(e, g)

            c_list = tuple(set(clique_list))
            for e in c_list:
                if e not in self.node_count:
                    self.node_count[e] = 1
                else:
                    self.node_count[e] += 1

            for e in c_list:
                h_g.add_edge(g, e + len(g_list), weight=(self.weight_vocab[(g, e)] / len(edges) + len(mcb_tuple)))
            for e in range(len(edges)):
                for i in range(e + 1, len(edges)):
                    for j in edges[e]:
                        if j in edges[i]:
                            weight = g_list[g].get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge = ((self.nodes_labels[edges[e][0] - 1], self.nodes_labels[edges[e][1] - 1]), weight)
                            weight_i = g_list[g].get_edge_data(edges[i][0], edges[i][1])['weight']
                            edge_i = ((self.nodes_labels[edges[i][0] - 1], self.nodes_labels[edges[i][1] - 1]), weight_i)
                            final_edge = tuple(sorted((self.add_to_vocab(edge), self.add_to_vocab(edge_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1
            for m in range(len(mcb_tuple)):
                for i in range(m + 1, len(mcb_tuple)):
                    for j in mcb_tuple[m]:
                        if j in mcb_tuple[i]:

                            weight = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring = [self.nodes_labels[mcb_tuple[m][t] - 1] for t in range(len(mcb_tuple[m]))]
                            cycle = (tuple(ring), weight)

                            weight_i = tuple(self.find_ring_weights(mcb_tuple[i], g_list[g]))
                            ring_i = [self.nodes_labels[mcb_tuple[i][t] - 1] for t in range(len(mcb_tuple[i]))]
                            cycle_i = (tuple(ring_i), weight_i)

                            final_edge = tuple(sorted((self.add_to_vocab(cycle), self.add_to_vocab(cycle_i))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1

            for e in range(len(edges)):
                for m in range(len(mcb_tuple)):
                    for i in edges[e]:
                        if i in mcb_tuple[m]:
                            weight_e = g_list[g].get_edge_data(edges[e][0], edges[e][1])['weight']
                            edge_e = ((self.nodes_labels[edges[e][0] - 1], self.nodes_labels[edges[e][1] - 1]), weight_e)
                            weight_m = tuple(self.find_ring_weights(mcb_tuple[m], g_list[g]))
                            ring_m = [self.nodes_labels[mcb_tuple[m][t] - 1] for t in range(len(mcb_tuple[m]))]
                            cycle_m = (tuple(ring_m), weight_m)

                            final_edge = tuple(sorted((self.add_to_vocab(edge_e), self.add_to_vocab(cycle_m))))
                            if final_edge not in self.edge_count:
                                self.edge_count[final_edge] = 1
                            else:
                                self.edge_count[final_edge] += 1

        return h_g
    def add_to_vocab(self, clique):
            c = copy.deepcopy(clique[0])
            weight = copy.deepcopy(clique[1])
            for i in range(len(c)):
                if (c, weight) in self.vocab:
                    return self.vocab[(c, weight)]
                else:
                    c = self.shift_right(c)
                    weight = self.shift_right(weight)
            self.vocab[(c, weight)] = len(list(self.vocab.keys()))
            return self.vocab[(c, weight)]

    def add_weight(self, node_id, g):
            if (g, node_id) not in self.weight_vocab:
                self.weight_vocab[(g, node_id)] = 1
            else:
                self.weight_vocab[(g, node_id)] += 1

    def update_weight(self, g):
            for (u, v) in g.edges():
                if u < len(self.data.g_list):
                    g[u][v]['weight'] = g[u][v]['weight'] * (math.log((len(self.data.g_list) + 1) / self.node_count[v - len(self.data.g_list)]))
                else:
                    g[u][v]['weight'] = g[u][v]['weight'] * (
                        math.log((len(self.data.g_list) + 1) / self.node_count[u - len(self.data.g_list)]))
            return g

    def add_edge(self, g):
            edges = list(self.edge_count.keys())
            for i in edges:
                g.add_edge(i[0] + len(self.data.g_list), i[1] + len(self.data.g_list), weight=math.exp(math.log(self.edge_count[i] / math.sqrt(self.whole_node_count[i[0]] * self.whole_node_count[i[1]]))))
            return g

    def drop_node(self, g):
            rank_list = []
            node_list = []
            sub_node_list = []
            for v in sorted(g.nodes()):
                if v > len(self.data.g_list):
                    rank_list.append(self.node_count[v - len(self.data.g_list)] / len(self.data.g_list))
                    node_list.append(v)
            sorted_list = sorted(rank_list)
            a = int(len(sorted_list) * 0.9)
            threshold_num = sorted_list[a]
            for i in range(len(rank_list)):
                if rank_list[i] > threshold_num:
                    sub_node_list.append(node_list[i])
            self.removed_nodes = sub_node_list
            count = 0
            label_mapping = {}
            for v in sorted(g.nodes()):
                if v in sub_node_list:
                    count += 1
                else:
                    label_mapping[v] = v - count
            for v in sub_node_list:
                g.remove_node(v)
            
            g = nx.relabel_nodes(g, label_mapping)
            return g

    @staticmethod
    def shift_right(l):
            if type(l) == int:
                return l
            elif type(l) == tuple:
                l = list(l)
                return tuple([l[-1]] + l[:-1])
            elif type(l) == list:
                return tuple([l[-1]] + l[:-1])
            else:
                print('ERROR!')

    @staticmethod
    def find_ring_weights(ring, g):
            weight_list = []
            for i in range(len(ring)-1):
                weight = g.get_edge_data(ring[i], ring[i+1])['weight']
                weight_list.append(weight)
            weight = g.get_edge_data(ring[-1], ring[0])['weight']
            weight_list.append(weight)
            return weight_list
