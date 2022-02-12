# HGETGI

A heterogeneous graph embedding model for predicting interactions between transcription factor and target gene

# Overview

- `data/` contains the necessary dataset files;
- `main.py` main function for HGETGI

# Requirement

The main requirements are:

- tqdm==4.60.0
- torch==1.8.1
- dgl==0.6.1
- numpy==1.19.5 

<p> To get the environment settled quickly, run: </p>

```
pip install -r requirements.txt
```

# Usage
The files in `data/`:
- "id_TF.txt": The id of transcription factor
- "id_Target.txt": The id of target gene
- "id_Disease.txt": The id of disease
- "TF_Target.txt": The interaction between transcription factor and target gene
- "TF_Disease.txt": The association between transcription factor and disease
- "Target_Disease.txt": The association between target gene and disease

Use *"main.py"* to train HGETGI model
```
python main.py
```
# HGETGI_EXE
The executable exe file of the HGETGI model has been uploaded to the releases directory, which can be downloaded and decompressed to predict the target genes corresponding to TF. First you should run the executable program and input the name of a TF you would like to query, and then input the number of target genes you would like to query. After the prediction, the target genes that ranking top *k* will be displayed and finally saved to the metapathTGI.csv file.


