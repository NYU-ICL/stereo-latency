#!/bin/bash

printf "\n*** Training baseline model ***\n"
python joint_model_train_agg.py

printf "\n*** Training ablation models ***\n"
python joint_model_train_agg.py --model_label vergence --lr 1e-2 --rng_seed 4
python joint_model_train_agg.py --model_label saccade --lr 1e-2

declare -a arr=(
    "Qi"
    "Ben"
    "Henry"
    "Esme"
    "Xi"
    "Andrew"
    "Kaisei"
    "Sophia"
)

for (( i = 0 ; i < ${#arr[@]} ; i++ )); do
    printf "\n*** Training models on data removal: uid ${arr[$i]} ***\n"
    python joint_model_train_agg.py --model_label "remove_${arr[$i]}" --uid_removal_mask "${arr[$i]}"
done

printf "\n*** Training models on data removal: random 8th ***\n"
python joint_model_train_agg.py --model_label "remove_random" --percent_removal_mask 0.125
