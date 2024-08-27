import esm
import torch
from DynaBCE.utils.pdb_utils import *
from DynaBCE.arg_parse import *


class phy_che_feat(PDBfuc):
    def __init__(self, pdbfile=None, res_phy_che=None):
        super().__init__(pdbfile)
        self.res_phy_che = res_phy_che
        self.pc_phy_che = []

    def get_phyche_feat(self):
        for res in self.get_sequence():
            self.pc_phy_che.append(self.res_phy_che[res])
        return np.stack(self.pc_phy_che).astype(float)

    def res_array(self):
        return self.get_phyche_feat()


class ESM2_feat(PDBfuc):
    def __init__(self, pdbfile=None, model_path=None):
        super().__init__(pdbfile)
        if model_path is not None:
            self.model_path = model_path
        else:
            print("Please download ESM2 model or path!")
        self.model, self.alphabet, self.batch_converter = self.load_model()

    def load_model(self):
        torch.hub.set_dir(self.model_path)
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model = model.eval()
        return model, alphabet, batch_converter

    def get_feat(self, pc):

        def save_seq_to_fasta(pc, sequence):
            file_path = f'{args.fasta_path}/{pc}.fasta'
            with open(file_path, 'w') as fasta_file:
                fasta_file.write(f'>{pc}\n{sequence}')

        data = [(pc, ''.join(self.get_sequence()))]
        save_seq_to_fasta(pc=data[0][0], sequence=data[0][1])

        _, _, batch_tokens = self.batch_converter(data)
        results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33][0][1:-1, :].detach().numpy()
        return token_representations


def get_seq_feat(args, pdbfile, pc):
    phy_che = np.loadtxt('./features/phyche_property.txt', dtype=str)
    res_phy_che = dict(zip(phy_che[1:, 0], phy_che[1:, 1:]))
    phyche = phy_che_feat(pdbfile, res_phy_che)
    phyche_feat = phyche.res_array()
    esm2 = ESM2_feat(pdbfile, args.esm_path)
    esm2_feat = esm2.get_feat(pc)
    return esm2_feat, phyche_feat

