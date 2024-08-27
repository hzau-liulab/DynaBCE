# Description
DynaBCE is dynamic ensemble algorithm to effectively identify conformational B-cell epitopes by leveraging feature- and template-based methods.  
![image](img/Framework.png)  

# Third-party software
GHECOM https://pdbj.org/ghecom/  
TM-align https://zhanggroup.org/TM-align/  
NW-align https://zhanggroup.org/NW-align/  
CD-HIT https://github.com/weizhongli/cdhit/releases  
DSSP https://swift.cmbi.umcn.nl/gv/dssp/DSSP_5.html  
NACCESS http://www.bioinf.manchester.ac.uk/naccess/  

# Database requirement
BCE633 
Manually created template library [Google Drive](https://drive.google.com/file/d/1hbQjtnSdU1I8TpVpdwHGx54xWhZheoGs/view?usp=share_link)

# Important python packages
Numpy  1.25.0
Pandas  1.2.0
Biopython  1.76
Scipy  1.10.1
cdhit-reader  0.1.1
fair-esm  2.0.0
pytorch  2.0.1
DGL  0.9.0
GraphRicciCurvature  0.5.3.1

# Usage
## 1. Download pre-trained models
The pre-trained models can be found at [Google Drive](https://drive.google.com/drive/folders/1TOp5xAqd5Wf_RpubCyrhouU_sX4FXLov?usp=sharing)
## 2. Configuration
Creat DynaBCE environment (conda env create -f environment.yaml).  
Manually download and install the third-party software listed above.  
Change the paths of these softwares and related databases at arg_parse.py 
Activate DynaBCE environment (conda activate NABind).  
## 3. Prediction
Run the following command:  

    python DynaBCE_model.py --pdb ./data/BCE633/7zyi_A.pdb --outdir ./output 

Type -h for help information:

    python DynaBCE_model.py -h

# Citation
Dynamic integration of feature- and template-based methods improves the prediction of conformational B-cell epitopes. *Submitted*, 2024.
