#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH -o output.txt
#SBATCH --job-name=cafin-nc

EPOCHS=100
NUM_RUNS=100
EXPS=(0 17 18 19)
DATASETS=("Cora" "CiteSeer" "EN" "Photo" "Computers")
SEED=0
TIME="`date +%d-%m-%H:%M`"

for DATASET in "${DATASETS[@]}"
do
    rm -rf res
    mkdir -p res
    mkdir -p res/$DATASET
    for i in "${EXPS[@]}"
    do
        mkdir -p res/$DATASET/EXP$i
        mkdir -p res/$DATASET/EXP$i/cls_acc
        mkdir -p res/$DATASET/EXP$i/con_mat
    done

    # centrality
    python centrality_measures.py --dataset $DATASET

    # graph_division
    python graph_division.py --dataset $DATASET

    # dist
    python approximate_distances.py --dataset $DATASET
    python dist.py --dataset $DATASET

    for i in "${EXPS[@]}"
    do
        for j in $(eval echo "{1..$NUM_RUNS}")
        do
            python train.py --dataset $DATASET --exp $i --run $j --epochs $EPOCHS --seed $SEED
            # Uncomment the below line and comment the line above to run CAFIN-AD
            # python train_approx.py --dataset $DATASET --exp $i --run $j --epochs $EPOCHS --seed $SEED
            python lr.py --exp $i --seed $SEED --dataset $DATASET
        done
    done

    python compare_imp.py --dataset $DATASET > res/$DATASET/out.txt
    mkdir -p res_log/$DATASET/$TIME
    cp -r res/$DATASET/* res_log/$DATASET/$TIME
done
