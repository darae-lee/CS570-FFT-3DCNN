#!/bin/bash

# Learning rates and lambda values to test
learning_rates=("0.0001" "0.005" "0.001" "0.01" "0.1" "0.2")
lambdas=("0.005" "0.001" "0.01" "0.1" "1" ""5"")

# Output directory for logs
log_dir="./logs"
mkdir -p $log_dir

# Run the script with each combination of learning rate and lambda
for lr in "${learning_rates[@]}"; do
    for lambda in "${lambdas[@]}"; do
        echo "Running with learning rate: $lr and lambda: $lambda"
        log_file="$log_dir/run_lr_${lr}_lambda_${lambda}.log"
        python run.py --dataset_dir kth-data-reg --add_reg --lr ${lr} --optim adam --lbda ${lambda}>${log_file} 2>&1
        echo "Finished running with learning rate: $lr and lambda: $lambda. Log saved to $log_file"
    done
done

echo "All runs completed."
