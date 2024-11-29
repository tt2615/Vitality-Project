#!/bin/bash

# Define parameter ranges
dims=(200 150 100 50 20)
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3)
batches=(64 32 16)
drop=(0.0 0.1 0.2 0.3 0.4 0.5)

# Iterate through parameter combinations
for dim in "${dims[@]}"; do
    for lr in "${lrs[@]}"; do
        for batch in "${batches[@]}"; do
            echo "Running with dim=$dim, lr=$lr, batch=$batch, drop=$drop"
            # python main.py --model=BertBpr_v3 --round=1 --device=cuda --dim=$dim --lr=$lr --batch=$batch
            python main.py --model=BertBpr_v3 --round=1 --comment=test_head2 --device=cuda --optim=Adam --dim=$dim --lr=$lr --batch=$batch --drop=$drop
            echo "---------------------------------------------------"
        done
    done
done 