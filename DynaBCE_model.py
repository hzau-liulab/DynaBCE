import sys
import os
proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)
from utils.model_utils import *


def loop(args, model, ml_model, pc_list, ag_ab, scaler, device, optimizer=None):
    all_probas, all_label, total_loss, pc_info = [], [], [], []
    for i in range(0, len(pc_list), args.batch_size):
        batch_pc = pc_list[i:i + args.batch_size]
        ml_batch, dl_batch, tm_batch, gating_input, batch_label, batch_res_type_id = \
            form_batch(args, batch_pc, ag_ab, scaler)
        ml_output = get_ml_output(ml_model, ml_batch)
        probas, expert_outputs, weights = model(dl_batch, tm_batch, gating_input, ml_output, device)
        loss = loss_fnc(probas, expert_outputs, batch_label.to(device), weights)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss.append(loss.item())
        probas = probas.detach().cpu().numpy()
        all_probas.append(probas.reshape(-1, 1))
        all_label.append(batch_label.reshape(-1, 1))
        pc_info.append(batch_res_type_id)
    return np.vstack(all_label), np.vstack(all_probas), round(np.mean(total_loss), 4), np.vstack(pc_info)


def model_train(args, ml_models, dl_models, tm_models, ag_ab, scaler, device):
    tr_probas, tr_true = [], []
    for fold in range(args.cv):
        print(fold)
        ML_model, DL_model, TM_model = load_fold_model(ml_models, dl_models, tm_models, fold, device)
        model = DynaBCE(DL_model, TM_model, tm_dim=256, input_dim=1820, hidden_dim=1024).to(device)
        optimizer = torch.optim.Adam([
            {'params': model.dl_model.mlp.parameters(), 'lr': args.dl_lr},
            {'params': model.tm_model.mlp_head.parameters(), 'lr': args.tm_lr},
            {'params': model.gating_network.parameters(), 'lr': args.lr, 'weight_decay': args.wd}
        ])
        tr_pc_list = get_pc_list('train', fold)
        val_pc_list = get_pc_list('val', fold)
        for epoch in range(args.num_epochs):
            model.train()
            _, _, _, _ = loop(args, model, ML_model, tr_pc_list, ag_ab, scaler, device, optimizer)

        model.eval()
        with torch.no_grad():
            val_label, val_probas, val_loss, _ = \
                loop(args, model, ML_model, val_pc_list, ag_ab, scaler, device)

        torch.save(model.state_dict(), f'{args.modules_path}/DynaBCE_{fold}.pth')
        tr_probas.extend(val_probas)
        tr_true.extend(val_label)
    return tr_probas, tr_true


def model_test(args, pc, ml_models, dl_models, tm_models, ag_ab, scaler, device, cutoff=0.25):
    with open(f'{args.modules_path}/DynaBCE_dict.pkl', 'rb') as fd:
        DynaBCE_models = pickle.load(fd)
    probas = []
    print(pc)
    for fold in range(args.cv):
        ML_model, DL_model, TM_model = load_fold_model(ml_models, dl_models, tm_models, fold, device)
        model = DynaBCE(DL_model, TM_model, tm_dim=256, input_dim=1820, hidden_dim=1024).to(device)
        model.load_state_dict(DynaBCE_models[fold])
        model.eval()
        with torch.no_grad():
            pc_label, pc_probas, pc_loss, pc_info = loop(args, model, ML_model, [pc], ag_ab, scaler, device)
        probas.append(pc_probas)
    probas_mean = np.mean(probas, axis=0)
    predicted_mean = np.where(probas_mean > cutoff, 1, 0)
    # pc_result = cal_metrics(predicted_mean, pc_label, probas_mean)
    # with open(f'{args.output_path}/DynaBCE_output.txt', 'a') as file0:
    #     print(f'===========  TR_test result, best_cutoff = {cutoff}  prob_mean  ===========\n'
    #           f'Recall: {result[0]}, Precision: {result[1]}, F1: {result[2]}, '
    #           f'AUC: {result[3]}, AUPR: {result[4]}, MCC: {result[5]}\n', file=file0)
    return np.hstack((predicted_mean.astype(int), probas_mean)), pc_info


def main():
    args = arg_parse()
    device = torch.device("cuda:%s" % args.gpuid if torch.cuda.is_available() else "cpu")
    set_random_seed(seed=2024)
    scaler, ag_ab = load_scaler_and_ag_ab()
    ml_models, dl_models, tm_models = load_modules(args)

    if args.train:
        # pc_list = np.loadtxt(f'./data/{args.dataset}.txt', dtype=str)
        tr_probas, tr_true = model_train(args, ml_models, dl_models, tm_models, ag_ab, scaler, device)
        best_cutoff, result_cv = model_evaluate(tr_probas, tr_true)

    if args.test:
        pc = os.path.basename(args.pdb).replace('.pdb', '')
        pred_result, pc_info = model_test(args, pc, ml_models, dl_models, tm_models, ag_ab, scaler, device)
        np.savetxt(f'{args.output_path}/{pc}.txt',
                   np.hstack((pc_info, pred_result)),
                   fmt='%s',
                   delimiter='\t',
                   header='\t'.join(['AA', 'Res_id', 'Binary', 'Proba']))


if __name__ == '__main__':
    main()
