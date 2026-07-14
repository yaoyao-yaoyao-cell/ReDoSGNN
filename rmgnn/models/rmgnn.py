import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
"""Neural components for the ReDoS-MotifGNN classifier.

The model combines embeddings from the motif meta-graph with local HRG graph
embeddings.  Names in this module follow the terminology used in the paper.
"""

from rmgnn.models.graph_encoder import GraphClassifier, GraphNodeClassifier
from rmgnn.models.heterogeneous_conv import WeightedGINConv
class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h

class Function(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Function, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

class Classifier(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type):
        super(Classifier, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = False

        self.classifierlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = Function(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = Function(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.classifierlayers.append(
                WeightedGINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(final_dropout)

    def forward(self, g, h, edge_weight):

        for i in range(self.num_layers - 1):
            h = self.classifierlayers[i](g, h, edge_weight)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            if i != 0:
                h = self.drop(h)
        return h

class StochasticClassifier(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type):
        super(StochasticClassifier, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = False

        self.classifierlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = Function(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = Function(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.classifierlayers.append(
                WeightedGINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(final_dropout)

    def forward(self, blocks, h, edge_weight):

        for i in range(self.num_layers - 1):
            h = self.classifierlayers[i](blocks[i], h, edge_weight[i])
            h = self.batch_norms[i](h)
            h = F.relu(h)
            if i != 0:
                h = self.drop(h)
        return h

class PredictionHead(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.head = nn.Linear(input_size, output_size)
        self.head2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, output):
        output = self.dropout(output)

        output = self.head(output)
        
        return output

class HighDropoutPredictionHead(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.head1 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.9)

    def forward(self, output):
        output = self.dropout(output)
        output = self.head1(output)
        return output

class RMGNNEncoderPair(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, pre_input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type, graph_pooling_type, device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.hidden_dim = hidden_dim
        self.graphclassifier = GraphClassifier(5, num_mlp_layers, pre_input_dim, 64, 64, 0.5, learn_eps, graph_pooling_type,
                                 neighbor_pooling_type, self.device).to(self.device)
        self.classifier = GraphNodeClassifier(num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, self.device).to(self.device)

class RMGNNClassifier(nn.Module):
    """Fuse motif meta-graph and local HRG representations for prediction."""

    def __init__(self, num_layers, num_mlp_layers, input_dim, pre_input_dim, hidden_dim,
                 output_dim, final_dropout, dropout_0, learn_eps,
                 neighbor_pooling_type, graph_pooling_type, device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.hidden_dim = hidden_dim
        self.heterogeneous_encoder = StochasticClassifier(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type).to(self.device)
        self.local_hrg_encoder = GraphClassifier(5, num_mlp_layers, pre_input_dim, 16, 16, dropout_0, learn_eps, graph_pooling_type, neighbor_pooling_type, self.device).to(self.device)
        self.prediction_head = PredictionHead(hidden_dim+16, output_dim).to(self.device)

    def forward(self, g, h, edge_weight, graph, num_cliques):
        local_embeddings = self.local_hrg_encoder(graph)
        motif_embeddings = self.heterogeneous_encoder(g, h, edge_weight)

        # Neighbor sampling may include auxiliary motif nodes that do not have
        # a corresponding input HRG. Pad only those trailing auxiliary rows.
        row_difference = motif_embeddings.shape[0] - local_embeddings.shape[0]
        if row_difference > 0:
            padding = torch.zeros(
                (row_difference, local_embeddings.shape[1]),
                device=local_embeddings.device,
            )
            local_embeddings = torch.cat([local_embeddings, padding], dim=0)
        elif row_difference < 0:
            local_embeddings = local_embeddings[: motif_embeddings.shape[0]]

        fused_embeddings = torch.cat((motif_embeddings, local_embeddings), dim=1)
        return self.prediction_head(fused_embeddings)


class ThreeClassifier(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, pre_input_dim1, pre_input_dim2, hidden_dim,
                 output_dim, final_dropout, dropout_0, learn_eps,
                 neighbor_pooling_type, graph_pooling_type, device=None):
        super(ThreeClassifier, self).__init__()
        self.device = torch.device(device) if device is not None else torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.hidden_dim = hidden_dim
        self.classifier1 = Classifier(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type).to(self.device)
        self.classifier2 = Classifier(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type).to(self.device)
        self.classifier3 = Classifier(num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps,
                 neighbor_pooling_type).to(self.device)
        self.graphclassifier1 = GraphClassifier(5, num_mlp_layers, pre_input_dim1, 16, 16, dropout_0, learn_eps, graph_pooling_type, neighbor_pooling_type, self.device).to(self.device)
        self.graphclassifier2 = GraphClassifier(5, num_mlp_layers, pre_input_dim2, 16, 16, dropout_0, learn_eps, graph_pooling_type, neighbor_pooling_type, self.device).to(self.device)
        self.head = PredictionHead(hidden_dim+16, output_dim).to(self.device)

    def forward(self, g, h, edge_weight, graph1, graph2, mask1, mask2):
        pre_h1 = self.graphclassifier1(graph1)
        pre_h2 = self.graphclassifier2(graph2)
        h = self.classifier1(g, h, edge_weight)
        h1 = torch.cat((h[mask1], pre_h1), 1)
        h1 = self.head(h1)
        h2 = torch.cat((h[mask2], pre_h2), 1)
        h2 = self.head(h2)
        return h1, h2

class ClassifierGraph(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        super(ClassifierGraph, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.classifierlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = Function(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = Function(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.classifierlayers.append(
                WeightedGINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.classifierlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer
