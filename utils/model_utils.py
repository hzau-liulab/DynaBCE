import random
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_curve, precision_recall_curve, auc, \
    matthews_corrcoef, precision_score, recall_score, f1_score
from DynaBCE.modules.model import *
from DynaBCE.features.sequence_feature import *
from DynaBCE.features.structure_feature import *
from DynaBCE.features.template_feature import *


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


def cal_metrics(predicted, y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    AUC = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    AUPR = auc(recall, precision)
    Precision = precision_score(y_true, predicted)
    MCC = matthews_corrcoef(y_true, predicted)
    Recall = recall_score(y_true, predicted)
    F1 = f1_score(y_true, predicted)
    return Recall, Precision, F1, AUC, AUPR, MCC


def model_evaluate(probas_val, true_val):
    cutoff_dict = {}
    for threshold_ in np.arange(0.00, 1.01, 0.01):
        predicted = []
        predicted_fold = [1 if proba > threshold_ else 0 for proba in probas_val]
        predicted.extend(predicted_fold)
        result_cv = cal_metrics(predicted, true_val, probas_val)
        cutoff_dict[round(threshold_, 2)] = list(result_cv)
    cutoff = max(cutoff_dict, key=lambda x: cutoff_dict[x][-1])
    result = cutoff_dict[cutoff]
    return cutoff, result


def loss_fnc(probas, outputs, target, weights):
    loss_fn = nn.BCELoss()
    dl_loss = loss_fn(outputs[:, 1], target)
    tm_loss = loss_fn(outputs[:, 2], target)
    gate_loss = loss_fn(probas, target)
    module_entropy = 0.01 * -torch.sum(weights * torch.log(weights + 1e-10), dim=1).sum()
    loss = dl_loss + tm_loss + gate_loss + module_entropy
    return loss


def get_ml_output(ml_model, batch_data):
    ml_out = ml_model.predict_proba(batch_data)
    ml_score = torch.tensor(ml_out[:, 1:2], dtype=torch.float32)
    return ml_score


def load_modules(args):
    with open(f'{args.modules_path}/ML_models_dict.pkl', 'rb') as fd:
        ml_models = pickle.load(fd)
    with open(f'{args.modules_path}/DL_models_dict.pkl', 'rb') as fd:
        dl_models = pickle.load(fd)
    with open(f'{args.modules_path}/TM_models_dict.pkl', 'rb') as fd:
        tm_models = pickle.load(fd)
    return ml_models, dl_models, tm_models


def load_fold_model(ml_models, dl_models, tm_models, fold, device):
    DL_model = BCEGVP(input_dim=1817, hidden_dim=1024, edge_dim=32).to(device)
    TM_model = ViT(image_size=(1, 540), patch_size=(1, 18), num_classes=1,
                   dim=256, depth=1, heads=4, mlp_dim=512, channels=6).to(device)
    ML_model = ml_models[fold]
    DL_model.load_state_dict(dl_models[fold])
    TM_model.load_state_dict(tm_models[fold])
    return ML_model, DL_model.eval(), TM_model.eval()


def form_batch(args, pc_list, ag_ab, scaler):
    ml_input, dl_input, tm_input, best_tm_input, label_list, res_type_id = [], [], [], [], [], []
    for i, pc in enumerate(pc_list):
        pc_label = np.loadtxt(f'{args.label_path}/{pc}.txt', dtype=object)
        pc_ml_input, pc_dl_input, pc_tm_input, pc_best_tm_feat, pc_res_type_id = \
            get_scaler_inputs(args, pc, ag_ab, scaler)
        ml_input.append(pc_ml_input)
        dl_input.append(pc_dl_input)
        tm_input.append(pc_tm_input)
        best_tm_input.append(pc_best_tm_feat)
        label_list.extend(pc_label[1:, 3].astype(int))
        res_type_id.append(pc_res_type_id)
    ml_batch = np.hstack(ml_input)
    dl_batch = Batch.from_data_list(dl_input)
    tm_batch = torch.cat(tm_input).unsqueeze(dim=2)
    best_tm_input = torch.cat(best_tm_input, dim=1)
    gating_input = torch.cat((best_tm_input, torch.tensor(ml_batch, dtype=torch.float32)), dim=1)
    batch_label = torch.tensor(label_list, dtype=torch.float32)
    batch_res_type_id = np.hstack(res_type_id)
    return ml_batch, dl_batch, tm_batch, gating_input, batch_label, batch_res_type_id


def get_scaler_inputs(args, pc, ag_ab, scaler):
    # pdbfile = f'{args.tm_library}TM_library_pdb/{tm_pc}.pdb'
    pdbfile = args.pdb
    pdb_fuc = PDBfuc(pdbfile)
    # with open(f'./data/7zyi_A.pkl', 'rb') as fd:
    #     ml_input, dl_input, tm_input, best_tm_feat = pickle.load(fd)

    res_type_id = np.array(pdb_fuc.res)[:, [0, 2]]
    esmif, hand_str_feat, res_distmap = get_str_feat(args, pdbfile, pdb_fuc, pc)
    esm2_feat, phyche_feat = get_seq_feat(args, pdbfile, pc)
    topk_tm_feat, best_tm_feat = get_template_feat(args, pdbfile, pdb_fuc, pc, ag_ab, scaler)

    phyche_str_feat = np.hstack((phyche_feat, hand_str_feat))
    phyche_str_feat[:, -25:-5] = scaler.transform(phyche_str_feat[:, -25:-5])
    scaler_feat = sigmoid(phyche_str_feat)

    ml_input = np.hstack((esm2_feat, esmif, scaler_feat))
    dl_input = featurize_as_graph(pc, pdb_fuc, res_distmap, ml_input)
    tm_input = torch.tensor(topk_tm_feat, dtype=torch.float32)
    best_tm_feat = torch.tensor(best_tm_feat, dtype=torch.float32)

    return ml_input, dl_input, tm_input, best_tm_feat, res_type_id


def load_scaler_and_ag_ab():
    with open('./data/ag_ab.pkl', 'rb') as fd:
        ag_ab = pickle.load(fd)
    with open('./features/scaler_dict.pkl', 'rb') as fs:
        scaler_dict = pickle.load(fs)
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = scaler_dict['scaler_mean'], scaler_dict['scaler_scale']
    return scaler, ag_ab


def get_pc_list(mode, fold=-1):
    if fold == -1:
        protein_chain = np.loadtxt(f'../data/{mode}.txt', dtype=str)
    else:
        protein_chain = np.loadtxt(f'..data/cv_split/cv_{mode}{fold}.txt', dtype=str)
    return protein_chain
