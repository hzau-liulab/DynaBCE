from DynaBCE.utils.pdb_utils import *
import subprocess
import joblib
import tempfile


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def NWalign(args, q_fasta, tm_fasta):
    def get_identity(out_file):
        return float(out_file.split('\n')[5].split()[2])

    out = subprocess.run([args.nwalign, q_fasta, tm_fasta], capture_output=True, check=True, text=True)
    NW_score = get_identity(out.stdout)
    return os.path.basename(tm_fasta).replace('.fasta', '.pdb'), NW_score


def TMalign(args, q_pdb, tm_pdb):
    def get_TMscore(out_file):
        tm1, tm2 = out_file.split('\n')[13:15]
        return float(tm1.split()[1]), float(tm2.split()[1])

    out = subprocess.run([args.tmalign, q_pdb, tm_pdb], capture_output=True, check=True, text=True)
    return os.path.basename(tm_pdb).replace('.pdb', ''), get_TMscore(out.stdout)


def topk_tm_search(args, pdbfile, pc, nw_cut=0.7, k=6):
    p = joblib.Parallel(n_jobs=-1)
    pc_idts = p(joblib.delayed(NWalign)(f'{args.fasta_path}/{pc}.fasta', f'{args.tm_library}/TM_library_fasta/{tm}')
                for tm in os.listdir(f'{args.tm_library}/TM_library_fasta/'))
    pc_idts = list(filter(lambda x: x[1] <= nw_cut, pc_idts))

    p = joblib.Parallel(n_jobs=-1)
    pc_tm_idts = p(joblib.delayed(TMalign)(pdbfile, f'{args.tm_library}/TM_library_pdb/{tm}')
                   for tm, nw_idt in pc_idts)
    sorted_pc_tm_idts = sorted(pc_tm_idts, key=lambda x: x[1][0], reverse=True)[:k]
    topk_tm = [(x[0], x[1][0]) for x in sorted_pc_tm_idts]
    return topk_tm


def AlignmentMap(args, pdbfile, pdb_fuc, tm, tm_pc_feat, scaler):
    def get_map_label(tm_output, tm_label_file):
        tm_lines = tm_output.split('\n')
        f_tm_label = np.loadtxt(tm_label_file, dtype=str)
        tm_label = f_tm_label[1:, 3].astype(int)
        map_label, map_feature = [], []
        tm_index = 0
        valid_chars = set(pdb_fuc.three_to_one.values())

        for s, char in enumerate(tm_lines[18].strip()):
            if char in valid_chars:
                if tm_lines[20][s] in valid_chars:
                    if tm_lines[19][s] == ':' or tm_lines[19][s] == '.':
                        map_label.append(tm_label[tm_index])
                        map_feature.append(tm_pc_feat[tm_index].reshape(1, -1))
                    else:
                        map_label.append(np.int64(0))
                        map_feature.append(np.zeros((1, tm_pc_feat.shape[1])))
                    tm_index += 1
                else:
                    map_label.append(np.int64(0))
                    map_feature.append(np.zeros((1, tm_pc_feat.shape[1])))
            elif tm_lines[20][s] in valid_chars:
                tm_index += 1
        pc_map_feature = np.vstack(map_feature)
        pc_map_feature[:, -25:-5] = scaler.transform(pc_map_feature[:, -25:-5])
        pc_map_feature[:, -25:] = sigmoid(pc_map_feature[:, -25:])
        return np.array(map_label).reshape(-1, 1), pc_map_feature

    cmd = [args.tmalign, pdbfile, f'{args.tm_library}/TM_library_pdb/{tm}.pdb']
    out = subprocess.run(cmd, capture_output=True, check=True, text=True)
    align_label, map_feature = get_map_label(out.stdout, f'{args.tm_library}/TM_library_label/{tm}.txt')
    return align_label, map_feature


def rotation_structure(args, pdbfile, pdb_fuc, tm):
    def get_rotation_matrix(rm_output):
        with open(rm_output, 'r') as frm:
            rm_lines = frm.read().split('\n')
        rotation_matrix = np.zeros((3, 4))
        for i, rm_line in enumerate(rm_lines[2:5]):
            rm = rm_line.split()
            for j in range(4):
                rotation_matrix[i, j] = float(rm[j + 1])
        return rotation_matrix

    with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_file:
        temp_file_path = temp_file.name

    def update_pdb_tempfile(original_pdb_file, new_coords):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.pdb') as temp_file:
            atom_index = 0
            for line in open(original_pdb_file, 'r'):
                if line.startswith(('ATOM', 'HETATM')):
                    x, y, z = new_coords[atom_index]
                    atom_index += 1
                    temp_file.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                else:
                    temp_file.write(line)
            temp_file_path = temp_file.name
        return temp_file_path

    cmd = [args.tmalign, pdbfile, f'{args.tm_library}/TM_library_pdb/{tm}.pdb', '-m', temp_file_path]
    subprocess.run(cmd, capture_output=True, text=True)
    rotation_matrix = get_rotation_matrix(temp_file_path)

    atom_coords = np.array(pdb_fuc.get_atom_coord_list())
    new_atom_coords_array = (rotation_matrix[:, 0].reshape(-1, 1) +
                             np.dot(rotation_matrix[:, -3:], atom_coords.T)).T
    new_atom_coords = new_atom_coords_array.tolist()
    rotation_pdbfile = update_pdb_tempfile(pdbfile, new_atom_coords)
    return rotation_pdbfile


def DistanceMap(rotation_pdb, ag_ab, tm_pc):
    tm_protein = tm_pc.split('_')[0]
    ab_chains = [ab.split('_')[1] for ab in ag_ab[tm_pc]]
    ab_pdbfile = f'./data/complex_pdb/{tm_protein}.pdb'
    bind = Binding(rotation_pdb, ab_pdbfile, ab_chains)
    distance_label = bind.res_array()[:, [1]].astype(float)
    return distance_label


def get_template_feat(args, pdbfile, pdb_fuc, pc, ag_ab, scaler):
    topk_tm = topk_tm_search(args, pdbfile, pc)
    topk_template_feat = []
    for tm_pc, tm_score in topk_tm:
        with open(f'{args.tm_library}/TM_library_feat/{tm_pc}.pkl', 'rb') as fd:
            tm_pc_feat = pickle.load(fd)
        align_label, align_feat = AlignmentMap(args, pdbfile, pdb_fuc, tm_pc, tm_pc_feat, scaler)
        rotation_pdbfile = rotation_structure(args, pdbfile, pdb_fuc, tm_pc)
        dist_label = DistanceMap(rotation_pdbfile, ag_ab, tm_pc)
        tm_scores = np.full((len(align_label), 1), tm_score)
        topk_template_feat.append(np.hstack((tm_scores, align_label, dist_label, align_feat)))
    stacked_feat = np.stack(topk_template_feat, axis=0)
    best_tm_feat = stacked_feat[0, :, :3]
    return np.transpose(stacked_feat, (1, 0, 2)), best_tm_feat
