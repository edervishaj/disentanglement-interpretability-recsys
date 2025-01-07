#!/bin/bash

MODELS="multidae multivae betavae"
DATASET="ML1M"
SEEDS="41 53 61 67 71"
EVALS="50"

for model in $MODELS; do
    for seed in $SEEDS; do
        # ./main.py prep-dataset $DATASET --min-rating-binarize 0.5 --validation-fraction 0.2 --seed $seed
        # ./main.py param-search $MODEL $DATASET --evals $EVALS --metrics ndcg disentanglement --seed $seed
        python main.py eval $model $DATASET --seed $seed --metrics ndcg recall --K 50 100
    done
done

