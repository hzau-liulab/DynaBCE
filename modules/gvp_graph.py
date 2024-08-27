import torch.sparse
import numpy as np
import math
import torch.nn.functional as F
import torch_geometric


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


class cal_feats:
    def __init__(self, num_positional_embeddings=16, num_rbf=16):
        super(cal_feats, self).__init__()
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

    def positional_embeddings(self, edge_index, num_embeddings=None):
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.)) / torch.tensor(num_embeddings))
        )

        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

    def dihedrals(self, X, eps=1e-7):
        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def rbf(self, D, D_min=0., D_max=20.):
        '''
        From https://github.com/jingraham/neurips19-graph-protein-design
        Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
        That is, shape [...dims],if `D` has  then the returned tensor will have
        shape [...dims, D_count].
        '''
        D_mu = torch.linspace(D_min, D_max, self.num_rbf)
        D_mu = D_mu.view([1, -1])
        D_sigma = torch.tensor((D_max - D_min) / self.num_rbf)
        D_expand = torch.unsqueeze(D, -1)

        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF


def get_coords(pdb_fuc):
    res_atom_coord = pdb_fuc.coord
    res_ids = res_atom_coord.keys()
    res_num = len(res_ids)
    coords = np.zeros((res_num, 4, 3))
    atoms = ["N", "CA", "C", "CB"]
    for i, res_id in enumerate(res_ids):
        for j, atom_type in enumerate(atoms):
            coord = res_atom_coord.get(res_id, {}).get(atom_type, np.zeros(3))
            coords[i, j] = coord
    return coords


def get_node_edge_feats(pdb_fuc, edge_index, node_feat):
    cal_feat = cal_feats()
    coords = get_coords(pdb_fuc)
    coords = torch.as_tensor(coords, dtype=torch.float32)
    coords_ca = coords[:, 1]
    pos_embeddings = cal_feat.positional_embeddings(edge_index)
    edge_vectors = coords_ca[edge_index[0]] - coords_ca[edge_index[1]]
    rbf = cal_feat.rbf(edge_vectors.norm(dim=-1))
    orientations = cal_feat.orientations(coords_ca)
    side_chains = cal_feat.sidechains(coords)

    edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
    edge_v = _normalize(edge_vectors).unsqueeze(-2)
    node_s = torch.tensor(node_feat, dtype=torch.float32)
    node_v = torch.cat([orientations, side_chains.unsqueeze(-2)], dim=-2)
    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
    return node_s, node_v, edge_s, edge_v


def get_edge_idx(res_distmap):
    adj = np.where(res_distmap > 10, 0, 1)
    adj_idx = np.argwhere(adj == 1)
    edge_index = torch.tensor(adj_idx.T, dtype=torch.long)
    return edge_index


def featurize_as_graph(pc, pdb_fuc, res_distmap, node_feat):
    '''
    node_s.shape, node_v.shape:  torch.Size([n, 1817]) torch.Size([n, 3, 3])
    edge_s.shape, edge_v.shape:  torch.Size([n, 32]) torch.Size([n, 1, 3])
    '''
    edge_index = get_edge_idx(res_distmap)
    node_s, node_v, edge_s, edge_v = get_node_edge_feats(pdb_fuc, edge_index, node_feat)
    pc_graph = torch_geometric.data.Data(name=pc, edge_idx=edge_index,
                                         node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v)
    return pc_graph
