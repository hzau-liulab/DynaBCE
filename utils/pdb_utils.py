import re
import os
import numpy as np
import itertools
import pandas as pd
from collections import Counter
import string
import pickle


class PDB(object):
    """
    NOTE: only single chain of pdb is accepted
    het_to_atom => dict contains het res to res and DNA res to one letter res (eg. DT => T)
    """
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hetatm.pkl'), 'rb') as f:
        het_to_atom = pickle.load(f)

    prorestypelist = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                      'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                      'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                      'SER', 'THR', 'TRP', 'TYR', 'VAL', ]

    three_to_one = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
                    'PHE': 'F', 'TRP': 'W', 'TYR': 'Y', 'ASP': 'D', 'ASN': 'N',
                    'GLU': 'E', 'LYS': 'K', 'GLN': 'Q', 'MET': 'M', 'SER': 'S',
                    'THR': 'T', 'CYS': 'C', 'PRO': 'P', 'HIS': 'H', 'ARG': 'R',
                    'UNK': 'X', 'MSE': 'M'}

    letter = string.ascii_uppercase

    def __init__(self, pdbfile=None, autofix=True, keephetatm=False, keepalternativeres=True, autocheck=False):
        """
        keephetatm: bool
        keepalternativeres: bool
        autocheck: bool (whether to remove residue without enough atoms)
        """
        self.res = []
        self.res_atom = []
        self.coord = {}
        self.coord_resatom = {}

        if isinstance(pdbfile, (np.ndarray, list)):
            self.pdb = pdbfile
        elif isinstance(pdbfile, str):
            self.pdb = open(pdbfile).readlines()
        else:
            return
        self._pdb_deal(keephetatm, keepalternativeres, autocheck)

        self._extraction()
        # self._xulie_deal()
        if len(self.res) != len(self.coord) and autofix:
            self._pdb_res_fix()
            self.res = list()
            self.res_atom = list()
            self.coord = dict()
            self.coord_resatom = dict()
            self._extraction()

        self.moleculer_type = self._check_type()
        self.het_to_atom = self.het_to_atom[self.moleculer_type]

        self.index_res = {i: j[-1] for i, j in enumerate(self.res)}
        self.res_index = {j[-1]: i for i, j in enumerate(self.res)}
        pass

    def _res(self, pdb_line):
        res_type = pdb_line[17:20].strip()
        chain = pdb_line[21:22].strip()
        res_id = pdb_line[22:28].strip()
        return res_type, chain, res_id

    def _coord(self, pdb_line):
        x = float(pdb_line[30:38].strip())
        y = float(pdb_line[38:46].strip())
        z = float(pdb_line[46:54].strip())
        return x, y, z

    def _atom(self, pdb_line):
        """
        Alternative also included, such as C3'A and C3'B KEEP C3'A
        """
        return pdb_line[12:16].strip()

    def _mix_info(self, pdb_line):
        Alter_id = pdb_line[16:17].strip()
        if Alter_id == '' or Alter_id == 'A':
            return self._res(pdb_line) + self._coord(pdb_line) + (self._atom(pdb_line),)
        else:
            return None

    def _pdb_deal(self, keephetatm=True, keepalternativeres=True, keepUNK=False, autocheck=False):
        """
        remove atom H
        """
        self.pdb = list(
            filter(lambda x: any((re.match('ATOM', x), re.match('HETATM', x))) and x.strip().split()[-1] != 'H',
                   self.pdb))
        if not keephetatm:
            self.pdb = self.remove_hetatm()
        if not keepalternativeres:
            self.pdb = self.remove_alternative()
        if not keepUNK:
            self.pdb = self.remove_UNK()
        if autocheck:
            self.pdb = self.correction()

    def _pdb_res_fix(self):
        """
        same res to add different letter eg. 60,60 => 60A,60B
        """
        reslist = [x[-1] for x in self.res]
        counter = Counter(reslist)
        element = list(filter(lambda x: x[1] > 1, counter.most_common()))
        for res, count in element:
            tmpindex = list(filter(lambda x: self.pdb[x][22:28].strip() == res, range(len(self.pdb))))
            tmpres_type = [self.pdb[i][17:20].strip() for i in tmpindex]
            tmpres_type2 = sorted(list(set(tmpres_type)), key=lambda x: tmpres_type.index(x))
            tmpdict = dict(zip(tmpres_type2, [' '] + list(self.letter[:len(tmpres_type2) - 1])))
            if len(tmpdict) != count:
                raise ValueError
            for i in tmpindex:
                pdbline = self.pdb[i]
                add = tmpdict[self.pdb[i][17:20].strip()]
                self.pdb[i] = pdbline[:22] + ' ' * (4 - len(res)) + res + add + pdbline[27:]

    def _check_type(self):
        """
        return: Protein/NA
        NOTE: when multiple chains pdb used, and chains are not in same moculer type, may lead error, CAUTIONS
        """
        restypelist = [x[0] for x in self.res]
        lenarray = np.array([len(x) for x in restypelist], dtype=int)
        leng = len(self.res) / 2
        if (lenarray == 3).sum() >= leng:
            return 'Protein'
        elif (lenarray < 3).sum() >= leng:
            return 'NA'
        else:
            return

    def remove_hetatm(self):
        """
        keep ATOM records only
        """
        return list(filter(lambda x: re.match('ATOM', x), self.pdb))

    def remove_alternative(self):
        """
        remove alternative residue eg. 12A
        """
        return list(filter(lambda x: re.search('\d$', x[22:28].strip()), self.pdb))

    def remove_UNK(self):
        """
        remove alternative residue eg. 12A
        """
        return list(filter(lambda x: x[17:20].strip() != 'UNK', self.pdb))

    def correction(self, num=1):
        """
        num: int
        if number of atoms in the res <= num, discard this res
        """
        self.pdb = list(self.pdb)
        rescount = dict()
        for i in range(len(self.pdb)):
            rescount.setdefault(self.pdb[i][22:28].strip(), [])
            rescount[self.pdb[i][22:28].strip()].append(i)
        delres = list(filter(lambda x: len(rescount[x]) < num, rescount))
        delindex = sum([rescount[x] for x in delres], [])
        index = [x for x in range(len(self.pdb)) if x not in delindex]
        return [self.pdb[x] for x in index]

    def _extraction(self):
        """
        self.res_atom: list => [[res_id,atomtype],......]
        self.coord: dict => {res_id:{atomtype:[x,y,z]},.....}
        self.coord_resatom: dict => {(x,y,z):[res_id,atomtype],.....}
        """
        # tmp_tuple = map(lambda x: _mix_info(x), self.pdb[3339:3358])
        tmp_tuple = filter(None, map(lambda x: self._mix_info(x), self.pdb))
        tmp_dict = {}
        for x in tmp_tuple:
            if [x[0], x[1], x[2]] not in self.res:
                self.res.append([x[0], x[1], x[2]])
            tmp_dict.setdefault(x[2], []).append((x[6], x[3:6]))
            self.coord_resatom[tuple(x[3:6])] = [x[2], x[6]]
            self.res_atom.append((x[2], x[6]))
        for key, value in tmp_dict.items():
            self.coord[key] = dict(value)

    def _coord_deal(self, mode='average'):
        """
        mode can be min, average, or atom name
        return dict => {res1:[average_coord(x,y,z)]}
        """
        res_coord = dict.fromkeys(self.coord.keys())
        if mode == 'min':
            for res in self.coord:
                res_coord[res] = list(map(lambda x: list(self.coord[res][x]),
                                          self.coord[res]))
            return res_coord

        elif mode == 'average':
            for res in self.coord:
                average_coord = []
                for atom, value in self.coord[res].items():
                    average_coord.append(list(value))
                average_coord = np.array(average_coord)
                res_coord[res] = [np.mean(average_coord, axis=0).tolist()]
            return res_coord

        else:
            # TODO in the statr or end of the pdb, there maybe some atom miss
            # what do we handle this
            # (for now, there is an error occure for this condition)
            for res_id in self.coord:
                if mode in self.coord[res_id]:
                    res_coord[res_id] = [list(self.coord[res_id][mode])]
                # if any(re.match(mode, x) for x in self.coord[res]):
                #     average_coord = []
                #     for atom, value in self.coord[res].items():
                #         if re.match(mode, atom):
                #             average_coord.append(list(value))
                #     average_coord = np.array(average_coord)
                #     res_coord[res] = [np.mean(average_coord, axis=0).tolist()]
                else:
                    average_coord = []
                    for atom, value in self.coord[res_id].items():
                        average_coord.append(list(value))
                    average_coord = np.array(average_coord)
                    res_coord[res_id] = [np.mean(average_coord, axis=0).tolist()]
                    # raise ValueError('check the input atom name (mode para)')
            return res_coord

    def get_res_coord_dict(self, mode='CA'):
        """
        mode can be average or atom name
        return dict => {res:[x,y,z],res:[x,y,z],.....}
        """
        res_coord = dict(map(lambda x: (x[0], np.squeeze(x[1]).tolist()),
                             self._coord_deal(mode).items()))
        return res_coord

    def get_res_coord_list(self, mode='CA'):
        res_coord = self.get_res_coord_dict(mode=mode)
        return np.array([res_coord[res] for _, _, res in self.res], dtype=float)

    def get_atom_coord_list(self):
        """
        return: ndarray of float (NO. of atoms * 3 (x,y,z))
        """
        return np.array(list(map(lambda x: self.coord[x[0]][x[1]], self.res_atom)))

    def get_atom_coord_dict(self):
        """
        return: dict => {atom:[x,y,z],.....}
        """
        return dict(map(lambda x: (x[1], np.squeeze(self.coord[x[0]][x[1]]).tolist()),
                        self.res_atom))

    def get_sequence(self, mode='oneletter'):
        if mode == 'oneletter':
            seq = [self.three_to_one[self.het_to_atom[x]] for x, _, _ in self.res]
        else:
            seq = [x for x, _, _ in self.res]
        return seq


def cal_distance(vec1, vec2):
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    return np.linalg.norm(vec1 - vec2)


def cal_angle(vec1, vec2):
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.arccos(np.clip(np.dot(vec1, vec2), -1., 1.))


class PDBfuc(PDB):
    def __init__(self, pdbfile, keephetatm=False, keepalternativeres=True, autocheck=False):
        super(PDBfuc, self).__init__(pdbfile, keephetatm=keephetatm, keepalternativeres=keepalternativeres,
                                     autocheck=autocheck)

    def res_distance(self, mode='CA'):
        """
        mode can be min, average, or atom name (eg. CA')
        min: the min distance
        average: the distance calcaluted by average coord
        atom: the distance calcaluted by atom coord
        return dict => {res1:(res2,distance),.......}
        """
        res_coord = self._coord_deal(mode)

        dim = len(self.index_res)
        index = np.where(np.triu(np.ones(shape=(dim, dim)), k=1))

        res_dist = dict()
        for i, j in zip(index[0], index[1]):
            res_i = self.index_res[i]
            res_j = self.index_res[j]
            coord_i = res_coord[res_i]
            coord_j = res_coord[res_j]
            dist = list(map(lambda x: cal_distance(x[0], x[1]),
                            itertools.product(coord_i, coord_j)))
            distance = min(dist)
            res_dist.setdefault(res_i, []).append((res_j, distance))
            res_dist.setdefault(res_j, []).append((res_i, distance))

        for _, res in self.index_res.items():
            res_dist.setdefault(res, []).append((res, 0.))

        return res_dist

    def distance_map(self, mode='CA'):
        """
        mode can be min, average, or atom name
        return array of float
        """
        dim = len(self.res_index)
        self.distmap = np.zeros((dim, dim))
        for key, value in self.res_distance(mode).items():
            index_i = self.res_index[key]
            for x, dist in value:
                index_j = self.res_index[x]
                self.distmap[index_i, index_j] = dist
        return self.distmap

    def angle_map(self, threeatoms=None):
        """
        threeatoms: list of atoms to decide the panel
        default ['CA','C','N'] for protein ["C3'","C1'","C5'"] for NucleicAcid
        return array of float
        """
        if threeatoms is None:
            threeatoms = ['CA', 'C', 'N'] if self.moleculer_type == 'Protein' else ["C3'", "C1'", "C5'"]

        def get_surface_normal(res):
            if all(x in self.coord[res] for x in threeatoms):
                vectors = np.array([self.coord[res][x] for x in threeatoms], dtype=float)
                normal = np.cross(vectors[1, :] - vectors[0, :], vectors[2, :] - vectors[0, :])
            elif len(self.coord[res]) == 1:
                normal = list(self.coord[res].values())[0]
            elif len(self.coord[res]) > 1:
                normal = np.mean(np.array(list(self.coord[res].values()), dtype=float), axis=0)
            else:
                raise ValueError(res)
            normal = np.array(normal, dtype=float)
            return normal

        ress = [x[-1] for x in self.res]
        anglemap = np.zeros((len(ress), len(ress)))
        for res1, res2 in itertools.product(ress, ress):
            i = self.res_index[res1]
            j = self.res_index[res2]
            if i == j:
                continue
            n1 = get_surface_normal(res1)
            n2 = get_surface_normal(res2)
            anglemap[i, j] = cal_angle(n1, n2)
        return anglemap

    def atom_distance(self):
        index = list(map(lambda x: '_'.join(x), self.res_atom))
        array = np.zeros(shape=(len(index), len(index)))
        for x, y in [(t1, t2) for t1 in range(len(index)) for t2 in range(t1 + 1, len(index))]:
            array[x, y] = cal_distance(
                self.coord[index[x].split('_')[0]][index[x].split('_')[1]],
                self.coord[index[y].split('_')[0]][index[y].split('_')[1]])
        array = array.T + array
        df = pd.DataFrame(array, index=index, columns=index)
        return df

    def contact_map(self, distmap=None, distcut=None):
        """
        return array of int
        self contact is not included
        """
        contmap = np.ones(distmap.shape, dtype=int)
        contmap[np.where(distmap > distcut)] = 0
        contmap[np.diag_indices_from(contmap)] = 0
        return contmap

    def contact(self, distmap_type='atom', discut=None, mode='min'):
        """
        if res contact: mode can be min, average, or atom name. eg. 'CA'
        else atom contact: distmap_df
        mode: min,average,atom name
        self.contactlist: [(res1,res2),(res9,res2),......]
        """
        if distmap_type == 'res':
            distmap = self.distance_map(mode=mode)
            distmap_index = self.index_res
            contact_map = self.contact_map(distmap, discut)
        else:
            distmap = self.atom_distance()
            distmap_index = distmap.index
            contact_map = self.contact_map(distmap.values, discut)
        contactlist = list()
        index_ij = np.nonzero(contact_map)
        for i, j in zip(index_ij[0], index_ij[1]):
            contact_pair = (distmap_index[i].split('_')[0], distmap_index[j].split('_')[0])  # distmap.columns
            if contact_pair[0] != contact_pair[1] and contact_pair not in contactlist:
                contactlist.append(contact_pair)
        return distmap, contactlist


class BindContact(PDBfuc):
    """
    NOTE: tarpdb and ligandpdb should to be single chain for now
    """
    def __init__(self, pdbfile=None, keephetatm=False, keepalternativeres=True, autocheck=False,
                 targetpdb=None, ligandpdb=None, dist_cutoff=4):
        super().__init__(pdbfile=pdbfile, keephetatm=keephetatm,
                         keepalternativeres=keepalternativeres, autocheck=autocheck)

        self.dist_cutoff = dist_cutoff
        self.tarpdb = targetpdb
        self.ligandpdb = PDB(ligandpdb)

        self.contactdict = dict()

        if all(x != None for x in (targetpdb, ligandpdb)):
            self._cal_contact()

        pass

    def _contact(self):
        for tarres, ligandres in itertools.product(self.tarpdb.res_index, self.ligandpdb.res_index):
            for taratom, ligandatom in itertools.product(self.tarpdb.coord[tarres], self.ligandpdb.coord[ligandres]):
                if self._cal_distance(self.tarpdb.coord[tarres][taratom],
                                      self.ligandpdb.coord[ligandres][ligandatom]) <= self.dist_cutoff:
                    self.contactdict.setdefault((tarres, ligandres), []).append((taratom, ligandatom))

    def _cal_distance(self, vec1, vec2):
        return cal_distance(vec1, vec2)

    def get_binding_res(self):
        """
        return binding res for tarpdb
        """
        self._cal_contact()
        return list(map(lambda x: x[0], self.contactdict))

    def get_contact_res(self):
        """
        return contact res pairs
        col 1 => tar res
        col 2 => ligand res
        NOTE: no ordered
        """
        self._cal_contact()
        return np.array(list(map(lambda x: list(x), self.contactdict)))

    def get_binding_res_atoms(self, mode='tar'):
        """
        get binding res atoms for tarpdb (mode=='tar') or for ligandpdb (mode=='ligand')
        """
        # TODO write
        pass

    def get_contact_res_atoms(self):
        """
        return dict => {(res1,res2):[(atom1,atom2),..],....}
        """
        self._cal_contact()
        return self.contactdict

    def _cal_contact(self):
        if not self.contactdict:
            self._contact()


class Binding(object):
    def __init__(self, tarpdb, ligandpdb, ligand_chains, dist_cutoff=4):
        """
        tarpdb: single chain pdb for binding res retrive, str(file)
        ligandpdb: single or more chains pdb, str(file)
        dist_cutoff: int/float
        """
        self.tarpdb = PDB(tarpdb)
        if isinstance(ligandpdb, str):
            with open(ligandpdb, 'r') as f:
                flist = f.readlines()
        else:
            flist = ligandpdb

        self.ligandpdblists = list()
        for chain in ligand_chains:
            self.ligandpdblists.append(list(filter(lambda x: x[21:22] == chain, flist)))

        self.dist_cutoff = dist_cutoff

    def get_binding_res(self):
        """
        return: list of binding res
        """
        bindingres = list()
        for ligandpdb in self.ligandpdblists:
            bindcontact = BindContact(targetpdb=self.tarpdb, ligandpdb=ligandpdb, dist_cutoff=self.dist_cutoff)
            bindingres.extend(bindcontact.get_binding_res())
        bindingres = set(bindingres)
        return list(bindingres)

    def res_array(self):
        """
        return: strarray col.1=>res col.2=>1/0
        """
        bindingres = self.get_binding_res()
        array = list()
        for res_type, chain, res in self.tarpdb.res:
            if res in bindingres:
                array.append([res, '1'])
            else:
                array.append([res, '0'])
        return np.array(array, dtype=str)
