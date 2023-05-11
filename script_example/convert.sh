# # convert dicom to nii.
# python convert.py \
# --convert_dataset_path /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/Dicom \
# --convert_dataset_saved /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/Nii \
# --convert_in_format dicom \
# --convert_out_format nii \
# --convert_dataset_depth 2 \
# --convert_dim 3 \
# --convert_tmp_keep \

# --convert_dataset_path /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/medical_image_process/example_image \
# --convert_dataset_saved /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/medical_image_process/tmp/nii \

# # extract from dicom with label a single image into png.
python convert.py \
--convert_dataset_path /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/Dicom \
--convert_dataset_saved /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/Nii \
--convert_in_format dicom \
--convert_out_format nii \
--convert_dataset_depth 2 \
--convert_dim 3 \
--convert_tmp_keep \
