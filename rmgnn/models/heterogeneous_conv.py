import torch as th
from torch import nn
import dgl.function as fn
from dgl.utils import expand_as_pair

class WeightedGINConv(nn.Module):
    """GIN-style propagation with optional normalized edge weights."""

    def __init__(
        self,
        apply_func=None,
        aggregator_type="sum",
        init_eps=0,
        learn_eps=False,
        activation=None,
    ):
        super().__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        if aggregator_type not in ("sum", "max", "mean"):
            raise KeyError(
                "Aggregator type {} not recognized.".format(aggregator_type)
            )
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", th.FloatTensor([init_eps]))

    def forward(self, graph, feat, edge_weight=None):

        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, _reducer("m", "neigh"))
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


# Compatibility alias for older experiment configurations.
SVC = WeightedGINConv
