# Subfolder Documentation

This folder contains transcriptome enrichment steps and further analysis.

## Notes

- A subfolder called "transcriptome" should contain the gencode version 45 transcripts and gtf annotations to be able to perform transcriptome methylation prediction.

- `retrieve_transcripts.ipynb` is a file that is needed to prepare the GENCODE v45 transcripts in an efficient manner for fast inference. In particular, it saves processed data as compressed NumPy archive (tx_data.npz)

- `inference_transcriptome.py` performs the actual prediction of the methylated cytosines on the transcriptome, given model weights and model configuration.

- `save_dataframe.ipynb` has been used to save the final dataframe with methylated cytosines across the transcriptome. The dataframe is the one offered to the scientific community.

- `secondary_structure.ipynb` is the file that has been used to predict the secondary structure of the predicted to be methylated cytosines on the transcriptome.

- `gene_enrichment.ipynb` has been used to perform the gene enrichment analysis of the predicted to be methylated transcripts.

