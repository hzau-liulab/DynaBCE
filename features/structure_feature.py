import esm.inverse_folding
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from DynaBCE.utils.pdb_utils import *
import scipy.stats as stats
import esm


class ESMIF_feat(PDBfuc):
    def __init__(self, pdbfile=None, chain_id=None, model_path=None):
        super().__init__(pdbfile)
        self.pdbfile = pdbfile
        if model_path is not None:
            self.model_path = model_path
        else:
            print("Please download ESM_IF1 model or path!")
        self.chain_id = chain_id
        self.model, self.alphabet = self.load_model()

    def load_model(self):
        # torch.hub.set_dir(self.model_path)
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        return model, alphabet

    def get_feat(self):
        structure = esm.inverse_folding.util.load_structure(self.pdbfile, self.chain_id)
        coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        esmif_feat = esm.inverse_folding.util.get_encoder_output(self.model, self.alphabet, coords)
        return esmif_feat.detach().numpy()


class RASA_feat(PDBfuc):
    def __init__(self, pdbfile=None, pc=None, max_asa=None, dssp_file=None, dssp_path=None):
        super().__init__(pdbfile)
        self.dssp_file = dssp_file
        self.dsspexe = dssp_path
        self.res_max_asa = max_asa
        # self.dssp(pdbfile)
        self.pc = pc

    def dssp(self, pdbfile=None):
        """
        excute dssp program
        """
        if pdbfile is not None:
            out_file = f'str_feat/dssp/{self.pc}.dssp'
            if os.path.exists(self.dsspexe):
                os.system(f'{self.dsspexe} -i {pdbfile} -o {out_file}')
                self.dssp_file = out_file
            else:
                print("Error: DSSP program is not installed. Please install dssp!")

    def rasa(self, dssp_file=None):
        """
        return: str_array col.1=>res col.2=>shellAcc col.3=>Rinacc col.4=>pocketness
        """
        res_rasa = []
        dssp_file = dssp_file if dssp_file is not None else self.dssp_file
        with open(dssp_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[28:]):
                aa = line[13]
                if aa == "!" or aa == "*" or aa == "X":
                    continue
                res_rasa.append(float(line[34:38].strip()) / self.res_max_asa[aa])
        return np.array(res_rasa).reshape(-1, 1)

    def res_array(self, dssp_file=None):
        return self.rasa(dssp_file)


def dp_feat(pc):
    dpfile = f'./features/STR_feature/DP/{pc}.tbl'
    if not os.path.exists(dpfile):
        print("Error: The tbl file does not exist. Please install PSAIA.")
    out = np.loadtxt(dpfile, skiprows=16, usecols=(18, 24), dtype=float)
    return out


class GE_feat(PDBfuc):
    def __init__(self, pdbfile=None, pc=None, ghecomresfile=None, ghecomexe=None):
        super().__init__(pdbfile)
        self.ghecomres = ghecomresfile
        self.ghecomexe = ghecomexe
        self.ghecom(pdbfile)

    def ghecom(self, pc, pdbfile=None):
        """
        excute ghecom program
        """
        if pdbfile is not None:
            ge_out = f'str_feat/GE/{pc}-ghe.txt'
            if os.path.exists(self.ghecomexe):
                os.system(f'{self.ghecomexe} -M M -atmhet B -hetpep2atm F -ipdb {pdbfile} -ores {ge_out}')
                self.ghecomres = ge_out
            else:
                print("Error: ghecomexe software is not installed. Please install ghecomexe!")

    def descriptor(self, ghecomresfile=None):
        """
        return: str_array col.1=>res col.2=>shellAcc col.3=>Rinacc col.4=>pocketness
        """
        ghecomres = ghecomresfile if ghecomresfile is not None else self.ghecomres
        out = np.loadtxt(ghecomres, skiprows=43, usecols=(3, 4, 7), dtype=float)
        out[:, 0] = out[:, 0] / 100
        out[:, 1] = out[:, 1]
        out[:, 2] = out[:, 2] / 100
        return out

    def res_array(self, ghecomresfile=None):
        return self.descriptor(ghecomresfile)


class CircularVariance(PDBfuc):
    """
    Reference: A new method for mapping macromolecular topography
    """

    def __init__(self, pdbfile, atom_distmap, r):
        """
        pdbfile: file
        r: int/float/list
        """
        super(CircularVariance, self).__init__(pdbfile)

        if isinstance(r, (int, float)):
            self.r = [r]
        elif isinstance(r, (list, tuple)):
            self.r = r
        else:
            raise ValueError('not accepted r')

        self.atom_distmap = atom_distmap
        self.atom_atom_vector = self._atom_atom_vector()

    def _atom_atom_vector(self):
        """
        return: dict {atom:{atom:vector},......}
        """
        atoms = self.atom_distmap.index.values
        atom_atom_vector = dict()
        for atom1 in atoms:
            atom_atom_vector[atom1] = \
                dict(map(lambda x: (x, np.array(self.coord['_'.join(atom1.split('_')[:-1])]
                                                [atom1.split('_')[-1]], dtype=float) -
                                    np.array(self.coord['_'.join(x.split('_')[:-1])]
                                             [x.split('_')[-1]], dtype=float)), atoms))
        return atom_atom_vector

    def _CVatom(self, r):
        """
        r: float
        return: dict {atom:cv,......}
        """

        def cv_calculate(atom, relative_atoms):
            vectors = np.array([self.atom_atom_vector[atom][x] for x in relative_atoms if x != atom], dtype=float)
            norms = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
            norm_vectors = vectors / norms
            return 1 - np.linalg.norm(np.sum(norm_vectors, axis=0)) / len(vectors)

        atoms = self.atom_distmap.index.values
        cv_atomdict = dict()
        for atom in atoms:
            dists = self.atom_distmap.loc[atom].values
            relative_atoms = atoms[np.where(dists < r)]
            cv_atomdict[atom] = cv_calculate(atom, relative_atoms)
        return cv_atomdict

    def _CVatom_array(self, r):
        """
        r: float
        return: ndarray (length of atoms*1)
        """
        cvatomdict = self._CVatom(r)
        return np.array(list(map(lambda x: cvatomdict['_'.join(x)], self.res_atom))).reshape(-1, 1)

    def _CVres(self, r):
        """
        r: float
        return: ndarray (length of res*1)
        """
        cvatomdict = self._CVatom(r)

        def cvcalculate(res):
            res_atoms = filter(lambda x: '_'.join(x.split('_')[:-1]) == res, cvatomdict)
            return np.mean([cvatomdict[x] for x in res_atoms])

        if r >= 100:
            cv_g = np.array([cvcalculate(x) for _, _, x in self.res], dtype=float).reshape(-1, 1)
            return stats.zscore(cv_g)
        else:
            return np.array([cvcalculate(x) for _, _, x in self.res], dtype=float).reshape(-1, 1)

    def CVatom(self):
        """
        return: ndarray (length of atoms*length of r)
        """
        return np.hstack(list(map(self._CVatom_array, self.r)))

    def CVres(self):
        """
        return: ndarray (length of res*length of r)
        """
        return np.hstack(list(map(self._CVres, self.r)))

    def atom_array(self):
        return self.CVatom()

    def res_array(self):
        return self.CVres()


class Graph(PDBfuc):
    def __init__(self, pdbfile, graph):
        super(Graph, self).__init__(pdbfile)
        """
        graph: edges (list/file)
        """
        if isinstance(graph, str):
            with open(graph, 'r') as f:
                edges = list(map(lambda x: x.strip().split(), f.readlines()))
        elif isinstance(graph, (list, np.ndarray)):
            edges = list(graph)
        else:
            raise ValueError('check edges in')

        self.g = nx.Graph()
        edges = list(map(lambda x: [self.res_index[y] for y in x], edges))
        self.g.add_edges_from(edges)


class RicciCurvature(Graph):
    def __init__(self, pdbfile, graph):
        """
        graph: edges (list/file)
        """
        super(RicciCurvature, self).__init__(pdbfile, graph)

        self.ORC = []
        self.FRC = []

    def ollivier_ricci(self, mode='sum'):
        """
        mode: str (sum/mean)
        return: dict
        """
        orc = OllivierRicci(self.g, alpha=0.5, verbose="INFO")
        orc.compute_ricci_curvature()
        g = orc.G
        aggregate = np.sum if mode == 'sum' else np.mean
        nodes = list(g.nodes)
        for n in self.index_res.keys():
            if n in nodes:
                curvature = list(map(lambda x: g[n][x]['ricciCurvature'], g[n]))
                curvature = aggregate(curvature)
                self.ORC.append(curvature)
            else:
                self.ORC.append(0)
        return np.array(self.ORC).reshape(-1, 1)

    def forman_ricci(self, mode='sum'):
        """
        mode: str (sum/mean)
        return: dict
        """
        frc = FormanRicci(self.g)
        frc.compute_ricci_curvature()
        g = frc.G
        aggregate = np.sum if mode == 'sum' else np.mean
        nodes = list(g.nodes)
        for n in self.index_res.keys():
            if n in nodes:
                curvature = list(map(lambda x: g[n][x]['formanCurvature'], g[n]))
                curvature = aggregate(curvature)
                self.FRC.append(curvature)
            else:
                self.FRC.append(0)
        return np.array(self.FRC).reshape(-1, 1)


class Topo(PDBfuc):
    def __init__(self, pdbfile, atom_contact):
        super(Topo, self).__init__(pdbfile)
        self.contact = atom_contact
        self.g = nx.Graph()

    def TPres(self):
        self.g.add_edges_from(self.contact)
        degrees = np.array(self.g.degree())[:, -1].astype(int)
        close = np.array(list(nx.closeness_centrality(self.g).values()))
        between = np.array(list(nx.betweenness_centrality(self.g).values()))
        clusters = np.array(list(nx.algorithms.clustering(self.g).values()))
        # return np.vstack((degrees, close, between, clusters)).T
        return np.vstack((degrees, stats.zscore(close), stats.zscore(between), clusters)).T

    def res_array(self):
        return self.TPres()


class Laplacian(PDBfuc):
    def __init__(self, pdbfile, res_distmap):
        super(Laplacian, self).__init__(pdbfile)
        self.distance_map = res_distmap

    def _sigmas(self):
        distmap_lower = np.tril(self.distance_map, k=-1)
        flattened_distance = distmap_lower[distmap_lower != 0]
        sigmas = np.percentile(flattened_distance, [0, 25, 50, 75, 100])
        return sigmas

    def _omega(self, sigma):
        dist = self.distance_map
        w = np.exp(-dist ** 2 / sigma ** 2)
        i, j = np.indices(w.shape)
        w[abs(i - j) <= 1] = 0
        sum_omega = np.sum(w, axis=1)
        return w, sum_omega

    def cal_laps(self, sigma):
        omega, sum_omega = self._omega(sigma)
        coords = np.array(list(self.get_res_coord_dict().values()))
        weighted_coords = np.dot(omega, coords)
        averaged_coords = weighted_coords / sum_omega.reshape(-1, 1)
        LAPS = np.linalg.norm(coords - averaged_coords, axis=1)
        if sigma in self._sigmas()[:-2]:
            return LAPS
        else:
            return stats.zscore(LAPS)

    def res_array(self):
        return np.array(list(map(lambda x: self.cal_laps(x), self._sigmas()))).T


class MultifractalDim(Graph):
    def __init__(self, pdbfile, graph):
        """
        graph: edges (list/file)
        """
        super(MultifractalDim, self).__init__(pdbfile, graph)

        self.MFD = list()

    def slope(self, weight=None):
        for index_res in self.index_res.keys():
            if index_res in self.g.nodes:
                self.MFD.append(self._slope(index_res, weight=weight))
            else:
                self.MFD.append(0)
        return np.array(self.MFD).reshape(-1, 1)

    def _slope(self, node, weight=None):
        m = nx.single_source_shortest_path_length if weight is None else nx.single_source_dijkstra_path_length
        spl = m(self.g, node)
        grow = [y for x, y in spl.items() if x != node]
        grow.sort()
        l_ml = [[x, y] for x, y in Counter(grow).items()]
        if len(l_ml) < 2:
            slope = 0
        else:
            l = np.log([x for x, y in l_ml])
            ml = np.log(np.cumsum([y for x, y in l_ml]))
            slope, intercept, r_value, p_value, std_err = stats.linregress(l, ml)
        return slope


def get_chain_id(pc, mode='native'):
    if mode == 'native':
        chain_id = pc.split('_')[1][:1]
    else:
        chain_id = 'A'
    return chain_id


def get_str_feat(args, pdbfile, pdb_fuc, pc):
    max_asa = np.loadtxt(f'./features/max_ASA.txt', dtype=str)
    res_max_asa = dict(zip(max_asa[:, 0], max_asa[:, 1].astype(int)))
    chain_id = get_chain_id(pc, mode='native')
    res_distmap, res_contact = pdb_fuc.contact(distmap_type='res', discut=7)
    atom_distmap, atom_contact = pdb_fuc.contact(discut=5)

    rasa_feat = RASA_feat(pdbfile, pc, res_max_asa, f'./features/STR_feature/DSSP/{pc}.dssp')
    rasa = rasa_feat.res_array()

    dp = dp_feat(pc)

    tp_feat = Topo(pdbfile, atom_contact)
    tp = tp_feat.res_array()

    mfd_feat = MultifractalDim(pdbfile, res_contact)
    mfd = mfd_feat.slope()

    rc_feat = RicciCurvature(pdbfile, res_contact)
    orc = rc_feat.ollivier_ricci()
    frc = rc_feat.forman_ricci()

    ghecom_feat = GE_feat(pdbfile, pc, ghecomexe=args.ghecom)
    ge = ghecom_feat.res_array(f'./features/STR_feature/GE/{pc}-ghe.txt')

    cv_feat = CircularVariance(pdbfile, atom_distmap, [12, 100])
    cv = cv_feat.res_array()

    ln_feat = Laplacian(pdbfile, res_distmap)
    ln = ln_feat.res_array()

    esmif_feat = ESMIF_feat(pdbfile, chain_id, args.esm_path)
    esmif = esmif_feat.get_feat()

    hand_str_feat = np.hstack((rasa, dp, tp[:, :1], tp[:, 3:], mfd, orc, frc, ge, cv[:, :1], ln[:, :-2],
                               tp[:, 1:3], cv[:, 1:], ln[:, -2:]))

    # hand_str_feat = np.column_stack((res_ids, np.hstack((rasa, dp, tp, mfd, cv, orc, frc, ge, ln))))

    return esmif, hand_str_feat, res_distmap
