# Example of single image report by image (mask) paths
# python report_generation/info_report.py \
# --report_image_path  /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/seg30/IMG/81580988-T1.nii.gz \
# --report_mask_path  /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/seg30/ANNO/81580988-T1.nii.gz \
# --report_describe 'consistent check' \


# # Example of multi image report by csv file.
# python report_generation/info_report.py \
# --report_image_path  ./output/paths_info.csv \
# --report_mask_col 'mask' \
# --image_type 'multi' \
# --report_describe 'Multi-image check' \

# Example of multi image report by image (mask) folder.
python report_generation/info_report.py \
--project_name ATLAS_LIVER_standard \
--report_image_path  /media/tx-deepocean/Data/workdir/wyd/2023/Segmentation/Dataset/nnUNet_raw/Dataset002_AtlasLiver/imagesTr \
--report_mask_path  /media/tx-deepocean/Data/workdir/wyd/2023/Segmentation/Dataset/nnUNet_raw/Dataset002_AtlasLiver/labelsTr \
--report_describe 'Multi-image check' \
