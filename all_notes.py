class TGNMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        aggr_module: Callable,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim

        self.aggr_module = aggr_module
        self.time_enc = TimeEncoder(time_dim)
        self.gru = GRUCell(msg_module.out_channels, memory_dim)

        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    def reset_parameters(self):
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        self.reset_state()

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            return self._get_updated_memory(n_id)
        else:
            return self.memory[n_id], self.last_update[n_id]

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst), then (dst -> src).
        msg_s, t_s, src_s, _= self._compute_msg(n_id, self.msg_s_store)
        msg_d, t_d, src_d, _= self._compute_msg(n_id, self.msg_d_store)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory, and then last_update.
        memory = self.gru(aggr, self.memory[n_id])
        last_update = scatter(t, idx, 0, dim_size=self.last_update.size(0), reduce="max")[n_id]

        return memory, last_update

    def _update_msg_store(
        self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, msg_store
    ):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id: Tensor, msg_store: TGNMessageStoreType):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0).to(self.device)
        dst = torch.cat(dst, dim=0).to(self.device)
        t = torch.cat(t, dim=0).to(self.device)
        # Filter out empty tensors to avoid `invalid configuration argument`.
        # TODO Investigate why this is needed.
        raw_msg = [m for i, m in enumerate(raw_msg) if m.numel() > 0 or i == 0]
        raw_msg = torch.cat(raw_msg, dim=0).to(self.device)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = torch.cat(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, src, dst

    def train(self, mode: bool = True):
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        argmax = scatter_argmax(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()


class LastNeighborLoader:
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size
        self.nbrs = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        self.reset_state()

    def __call__(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        nbrs = self.nbrs[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        # Filter invalid nbrs (identified by `e_id < 0`).
        mask = e_id >= 0
        nbrs, nodes, e_id = nbrs[mask], nodes[mask], e_id[mask]

        # Relabel node indices.
        n_id = torch.cat([n_id, nbrs]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        nbrs, nodes = self._assoc[nbrs], self._assoc[nodes]

        return n_id, torch.stack([nbrs, nodes]), e_id

    def insert(self, src: Tensor, dst: Tensor):
        # Collect central nodes, their nbrs and the current event ids.
        nbrs = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(
            self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
        ).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        nbrs, e_id = nbrs[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_nbrs = e_id.new_empty(n_id.numel() * self.size)
        dense_nbrs[dense_id] = nbrs
        dense_nbrs = dense_nbrs.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        nbrs = torch.cat([self.nbrs[n_id, : self.size], dense_nbrs], dim=-1)

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.nbrs[n_id] = torch.gather(nbrs, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class TransformerConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int,
        dropout: float = 0.0
    ):
        self.aggr_module = SumAggr()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)
        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_skip = Linear(in_channels, heads * out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor):
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)

        j, i = edge_index[0], edge_index[1] # i is the receiver, j is the sender
        out = self.message(query[i], key[j], value[j], edge_attr, index=i, size_i=query.size(0))
        out = self.aggr_module(out, i, dim=0, dim_size=query.size(0))

        out = out.view(-1, self.heads * self.out_channels)
        return out + self.lin_skip(x)

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        size_i: int
    ) -> Tensor:
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr=None, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        out = out + edge_attr
        out = out * alpha.view(-1, self.heads, 1)
        return out


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

# ---------------------------------------------------------------------------

memory_dim = time_dim = embedding_dim = 100

neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    aggr_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(in_channels=memory_dim, out_channels=embedding_dim, msg_dim=data.msg.size(-1), time_enc=memory.time_enc).to(device)
decoder = LinkPredictor(in_channels=embedding_dim).to(device)

opt = Adam(
    set(memory.parameters()) | set(gnn.parameters()) | set(decoder.parameters()),
    lr=0.0001,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    memory.train()
    gnn.train()
    decoder.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    for batch in train_loader:
        opt.zero_grad()

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
        pos_out = decoder(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = decoder(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        opt.step()
        memory.detach()

@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    decoder.eval()

    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])
        decoder(z[assoc[batch.src]], z[assoc[batch.dst]])
        decoder(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

