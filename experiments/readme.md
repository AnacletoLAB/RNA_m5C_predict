# Subfolder Documentation

This folder contains all the scripts for the experiments, including: cross-validation and grid search, hard negative mining, model training, transcriptome predictions and enrichment, model comparison and surrogate model training. I will briefly summarize what each script and folder was used for. Several subfolder contain further "Readme" to help you understand better the structure of the repository.

## Scripts

- `analysis_atlas.py`: This script was utilized to run the transformer model and the position frequency matrices over the m6A Atlas to check for true methylated sites vs false positive calls.

- `analysis_grid_search.py`: This script was utilized to analyze the resulting performances of Bi-GRU, Transformer Encoder and 1D-CNN during hyper-parameters grid search and 5-Fold cross-validation.

- `analysis_motifs.ipynb`: This script was utilize to analyze the motifs that were wrongly predicted by the trained models. The visualization of the confusion matrices showed us that the models were making mistakes on few "hard negatives", and prompted us to perform the "hard negatives" mining step.

- `analysis_performance_binary.ipynb`: This script was utilize to check the performance of the binary classifiers during the section "model comparison" of the article.

- `inference.py`: This script was utilized to perform the inference on the dataset by the trained models. Results were analyzed by `analysis_motifs.ipynb`.

- `nucleotide_composition.ipynb`: This script was utilized to check the nucleotide compositions of the different positive and negatives across dataset in the section "model comparison" of the article. Jensen-Shannon divergence was calculated and its part of the results of the article.

- `run_mavenn.ipynb`: This script was utilized to train surrogate models with the MAVENN Python package.

- `train_best.py`: This script was utilized to train the three best architectures after grid search and 5-Fold cross-validation on the whole training set for Bi-GRU, Transformer Encoder and 1-D CNN.

- `train_grid_pool.py`: This script was utilized to run on multiple processes the grid search.

- `train_lgbm.py`: This script was utilized to train the LightGBM with TNC encodings by the authors of Mlm5C

- `train_one_run.py`: This script is called by `train_grid_pool.py` for parallel training.

- `train.py`: This is just a training script that was utilized to perform several trial and error runs to pin down reasonable hyper-parameters for the grid search and other training steps.

## Folders

- data_mlm5c: to put the Mlm5C Dataset for model comparison (further Readme inside folder specifies more).

- dataset_generation: This script contains tokenizer and "pytorch" dataset generation modules, needed to convert the data to pytorch datasets, needed to oerform model training and inference. 

- deepm5C_DataSet: to put the Deep5mC Dataset for model comparison (further Readme inside folder specifies more).

- hard_negatives: This folder contains all the analysis performed for "hard negative" mining and training augmentation (further Readme inside folder specifies more).

- models: This folder contains all models' architectures, including Bi-GRU, 1D-CNN and Transformer encoder modules.

- models_params: The .json files here contain the best hyper-parameters for Bi-GRU, 1D-CNN and Transformer encoder after grid search.

- transcriptome_analysis: This folder contains all the analysis performed on the transcriptome: predicted methylations on the human transcriptome cytosines, methylated sites motifs and secondary structures, gene enrichment etc.  (further Readme inside folder specifies more).

- utils: Important scripts divided in a subfolder to avoid cluttering the working directory.