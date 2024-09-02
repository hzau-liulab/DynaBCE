# Description
DynaBCE is dynamic ensemble algorithm to effectively identify conformational B-cell epitopes by leveraging feature- and template-based methods. Using novel structural descriptors and embeddings from language models, we built machine learning and deep learning modules based on boosting algorithms and geometric graph neural networks, respectively. Meanwhile, we created a template module by combining similar antigen structures and transformer-based algorithms to capture antibody binding signatures. Finally, we designed a gating network to dynamically assign the weights of three modules for each residue to yield its integrative prediction result.   
![image](img/Framework.png)  

# Usage
## Installation 
1. Clone the repository to your local device.
   ```shell
    git clone https://github.com/hzau-liulab/DynaBCE   
    cd DynaBCE
   ```
2. Install the necessary dependencies.     
   ** Python packages         
        python                3.9.17    
        Numpy                 1.26.4     
        Pandas                2.1.2    
        Biopython             1.79     
        Scipy                 1.11.4      
        fair-esm              2.0.1      
        pytorch               1.12.1    
        vit-pytorch           1.6.4     
        pyg                   2.3.1      
        scikit-learn          1.3.2    
        GraphRicciCurvature   0.5.3.1 

     
   We recommend creating a new conda environment for DynaBCE, and then install the required packages within this environment.          
   ```shell
    conda env create -f environment.yaml  
    conda activate DynaBCE
   ```
    ** Third-party software        
        DSSP https://swift.cmbi.umcn.nl/gv/dssp/    
        PSAIA https://static-content.springer.com/esm/art%3A10.1186%2F1472-6807-8-21/MediaObjects/12900_2007_186_MOESM20_ESM.bin        
        GHECOM https://pdbj.org/ghecom/      
        TM-align https://zhanggroup.org/TM-align/   
        NW-align https://zhanggroup.org/NW-align/
   
   Manually download and install the third-party software listed above. Please place all softwares in the `./software` directory.
   
4. Download database and pre-trained models        
   BCE633 dataset and Manually created template library [Dataset](https://drive.google.com/file/d/1z1xSP5U5GkCvLTmrMAnlxp8qUMspBr9y/view?usp=sharing)      
   Our pre-trained models can be found at [DynaBCE model](https://drive.google.com/file/d/1z1xSP5U5GkCvLTmrMAnlxp8qUMspBr9y/view?usp=sharing)         
   ESM-2 [esm2_t33_650M_UR50D](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)        
   ESM-IF1 [esm_if1_gvp4_t16_142M_UR50](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)            

Please place the relevant databases in the `./data` directory. Pre-trained models should be placed in the `./modules` directory. ESM-2 and ESM-IF1 should be placed in the `./esm_model` directory. Alternatively, you can change the relevant paths in `arg_parse.py.`

## Run DynaBCE model  
1. Prepare input PDB file.      
   Each input file should be saved in a separate PDB file named `protein_chain.pdb`.       
   
2. Run the prediction    
   Run the following command:   
   
       python DynaBCE_model.py --pdb ./data/BCE633/7zyi_A.pdb --fasta_path ./data/BCE633_fasta --ghecom ./software/ghecom --dssp ./software/dssp --esm_path ./esm_model --tmalign ./software/TMalign --nwalign ./software/NWalign --tm_library ./data  --modules_path ./modules --output_path ./output --test True
   
   Type -h for help information:         
   
       python DynaBCE_model.py -h         

# Citation
Dynamic integration of feature- and template-based methods improves the prediction of conformational B-cell epitopes. *Submitted*, 2024.
