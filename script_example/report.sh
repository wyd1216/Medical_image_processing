#python info_report.py \
#--image_path  /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/Nii/T2/81503082_T2.nii.gz \
#--report_title "Medical Image Report" \
#--report_pdf report_tmp0.pdf \
#--image_type 'nii' \
#--report_describe 'consistent check' \
#--report_tmp_keep \

#python info_report.py \
#--image_path  /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/monai_mt/T2/82703531_T2.nii.gz \
#--report_title "Medical Image Report" \
#--report_pdf report_tmp1.pdf \
#--image_type 'single' \
#--report_describe 'consistent check' \
#--report_tmp_keep \

#python info_report.py \
#--image_path  /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/fixed_transforms/t1+c/82639170_T1.nii.gz \
#--report_title "Medical Image Report" \
#--report_pdf report_tmp2.pdf \
#--image_type 'single' \
#--report_describe 'consistent check' \
#--report_tmp_keep \

#python info_report.py \
#--image_path  /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/fixed_transforms/T2/80113753_T2.nii.gz \
#--report_title "Medical Image Report" \
#--report_pdf report_tmp3.pdf \
#--image_type 'single' \
#--report_describe 'consistent check' \
#--report_tmp_keep \

#python info_report.py \
#--image_path /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/t1+c/80113753_T1 \
#--report_title 北大人民四肢骨肿瘤 \
#--report_pdf report_t1_80113753.pdf \
#--report_describe 'MRI dataset T1CE'

#python info_report.py \
#--image_path /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/t1+c/82639170_T1 \
#--report_title 北大人民四肢骨肿瘤 \
#--report_pdf report_t1_82639170.pdf \
#--report_describe 'MRI dataset T1CE'

#python info_report.py \
#--image_path /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/T2/80113753_T2 \
#--report_title 北大人民四肢骨肿瘤 \
#--report_pdf report_t2_80113753.pdf \
#--report_describe 'MRI dataset T2'

#python info_report.py \
#--image_path /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/T2/82639170_T2 \
#--report_title 北大人民四肢骨肿瘤 \
#--report_pdf report_t2_82639170.pdf \
#--report_describe 'MRI dataset T1CE'

# Example of multi image report
python info_report.py \
--image_path  /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/convert_data_info.csv \
--report_title "Medical Image Report" \
--report_pdf report_convert_multi.pdf \
--image_type 'multi' \
--report_describe 'Multi-image check' \
--report_tmp_keep \

#python info_report.py \
#--image_path  /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/preprocess_info.csv \
#--report_title "Medical Image Report" \
#--report_pdf report_preprocess_multi.pdf \
#--image_type 'multi' \
#--report_describe 'Multi-image check' \
#--report_tmp_keep \

#python info_report.py \
#--image_path  /media/tx-deepocean/Data/workdir/wyd/2023/BeiDaRenMin/Osteosarcoma/Dataset/preprocess_info_test.csv \
#--report_title "Medical Image Report" \
#--report_pdf report_preprocess_multi_test.pdf \
#--image_type 'multi' \
#--report_describe 'Multi-image check' \
#--report_tmp_keep \
