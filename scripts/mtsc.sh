#! /bin/bash

codebook_size=$1
dim_model=$2

python ./benchmarks/mtsc.py \
    --checkpoint_dir "./checkpoints" \
    --dataset_dir "/path/to/UEA_multivariate" \
    --save_dir "./results" \
    --checkpoint_name "uea_dim${dim_model}_codebook${codebook_size}" \
    --mode all \
    --feature_type all \
    --validation default