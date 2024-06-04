RAW_SCANNET_DIR=/home/nibio/mutable-outside-world/code/Pointcept/forest/test_plots
PROCESSED_SCANNET_DIR=/home/nibio/mutable-outside-world/code/Pointcept/forest/test_plots_processed

# python3 pointcept/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}

# train on scannet
sh scripts/train.sh -g 8 -d scannet -c pretrain-msc-v1m1-0-spunet-base_forest -n pretrain-msc-v1m1-0-spunet-base_forest



# ScanNet20 Semantic Segmentation
# sh scripts/train.sh -g 8 -d scannet -w exp/scannet/pretrain-msc-v1m1-0-spunet-base/model/model_last.pth -c semseg-spunet-v1m1-4-ft -n semseg-msc-v1m1-0f-spunet-base

# ScanNet20 Instance Segmentation (enable PointGroup before running the script)
# sh scripts/train.sh -g 8 -d scannet -w exp/scannet/pretrain-msc-v1m1-0-spunet-base/model/model_last.pth -c insseg-pointgroup-v1m1-0-spunet-base -n insseg-msc-v1m1-0f-pointgroup-spunet-base
