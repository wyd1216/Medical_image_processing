#python fixed_transforms.py \
#--mt_save_root_replace Nii monai_mt \
#--preprocess_info_savepath /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/preprocess_info_test.csv \
#--preprocess_info_input /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/all_info2.csv \
#--preprocess_resample 1.0 1.0 5.0 \

python manipulation/fixed_transforms.py \
--mt_save_root_replace Nii fixed_transforms \
--preprocess_info_input /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/convert_data_info.csv \
--preprocess_info_savepath /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/preprocess_info.csv \
--preprocess_resample 1.0 1.0 5.0 \
