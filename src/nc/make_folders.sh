DATASETS=("Cora" "CiteSeer" "EN" "Photo" "Computers" "ppi")
EXPS=(0 17 18 19)

for DATASET in "${DATASETS[@]}"
do
    mkdir -p res
    mkdir -p res/$DATASET
    mkdir -p res/$DATASET/imp
    mkdir -p res/$DATASET/class_acc
    mkdir -p res/$DATASET/overall_acc
    mkdir -p res/$DATASET/overall_f1
    mkdir -p res/$DATASET/imp_f1
    for i in "${EXPS[@]}"
    do 
        mkdir -p res/$DATASET/EXP$i
        mkdir -p res/$DATASET/EXP$i/cls_acc
        mkdir -p res/$DATASET/EXP$i/con_mat
    done
done