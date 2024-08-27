import argparse


def arg_parse():
    parser = argparse.ArgumentParser("get features setting")
    parser.add_argument("--gpuid", type=int, default=0, help="")
    parser.add_argument("--pdb", type=str, default='./data/BCE633/7zyi_A.pdb', help='input pdb')
    parser.add_argument("--fasta_path", type=str, default='./data/BCE633_fasta', help="")
    parser.add_argument("--label_path", type=str, default='./data/BCE633_label', help="")
    parser.add_argument("--dataset", type=str, default='./data/Test_56.txt', help="")
    parser.add_argument("--modes", type=list, default=['train', 'val', 'test'], help="dataset split")
    parser.add_argument("--ghecom", type=str, default='/DynaBCE/software/ghecom/ghecom', help="ghecom software")
    parser.add_argument("--dssp", type=str, default='/home/yyShen/.conda/envs/test/bin/mkdssp', help="dssp software")
    parser.add_argument("--esm_path", type=str, default='/8Tdata/esm_model', help="esm model path")
    parser.add_argument("--tmalign", type=str, default='/home/yyShen/TMalign', help="TMalign software")
    parser.add_argument("--nwalign", type=str, default='/home/yyShen/Software/NWalign', help="NWalign software")
    parser.add_argument("--tm_library", type=str, default='./data', help="tm library data")
    parser.add_argument("--modules_path", type=str, default='./modules', help="three pre-train modules")
    parser.add_argument("--output_path", type=str, default='./output', help="output file")
    parser.add_argument("--cv", type=int, default=10, help="cross validate")
    parser.add_argument("--num_epochs", type=int, default=15, help="model epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-8, help="model learning rate")
    parser.add_argument("--dl_lr", type=float, default=2e-6, help="dl module learning rate")
    parser.add_argument("--tm_lr", type=float, default=2e-5, help="tm module learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="L2 regularization parameter")
    parser.add_argument('--train', type=bool, default=False, help="train a model")
    parser.add_argument('--test', type=bool, default=True, help="test a trained model")
    return parser.parse_args()
