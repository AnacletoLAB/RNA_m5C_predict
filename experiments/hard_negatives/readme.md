# Subfolder Documentation

This folder contains the hard negatives mining and fine-tuning steps.

## Notes

The parameters now are selected for the Bi-GRU model in the article. In particular, in each file `model_name = RNN`. One can change it to `1DCNN` or `Transformer` to run the same experiments for the other two models.

## How to Use

- Start by running `train.py`. This will train the 5 fold initial models.

- Run `inference_all_negatives.py` to predict labels on the whole remaining negatives non-selected for training and test set.

- Now that you have predictions for all remaining negatives for all models (1 per fold), you can run `hard_negatives_augmentation.py` by selecting a list of upper and lower bounds to filter the hard negatives based on the probability the model assigns to them. Right now, the `quota`, namely the number of hard negatives per class is set to:  

  $\text{Negatives Added Per Class} = \frac{\text{N Negs Training}}{\text{Num Classes}}$  

  If there are more negatives with such bounds, otherwise, they are all selected. Importantly, the script filter sequences with no more than $90\%$ sequence identity within the central 21 nucleotides with both the full dataset as well as other negatives previously selected to avoid data leakage.

- Once the hard negatives are generated, one can run the script `run_all_bounds.sh` to run in parallel the fine-tuning steps with hard-negatives selected with different upper bounds.

- Results are visualized with the notebook `data_analysis.ipynb`.

- All the scripts that end with `final` have been used to perform the final hard negatives mining and fine-tuning for the model trained on the whole training set.

- All the the scripts that end with `final_hard` have been used to perform the final more harsh mining and fine-tuning for the model trained on the whole training_set