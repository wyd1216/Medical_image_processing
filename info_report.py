import argparse
import shutil
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from reportlab.platypus import Table, SimpleDocTemplate, Spacer
from reportlab.lib.pagesizes import letter  # 页面的标志尺寸(8.5*inch, 11*inch)

from config import get_args_parser
from image_processing import MedicalImageProcessor
from report_generation import Graphs

def report_multi_image(args):
    info_df = pd.read_csv(args.image_path)
    tmp_dir = Path('./tmp')
    # Get the current time
    current_time = datetime.now()
    # Format the current time as "YYYYMMDD"
    formatted_time = current_time.strftime("%Y%m%d")
    tmp_dir = tmp_dir / formatted_time
    tmp_dir.mkdir(exist_ok=True, parents=True)
    saved_dir = Path(args.report_saved_dir)
    # ----------------------report information generate----------------------------
    content = list()
    # 创建内容对应的空列表
    content.append(Graphs.draw_title(args.report_title))
    if args.report_describe:
        content.append(Graphs.draw_text(args.report_describe))

    for index, row in tqdm(info_df.iterrows(), total=info_df.shape[0], desc="Processing rows"):
        image_path = row[args.report_image_col]
        image_name = Path(image_path).with_suffix('').with_suffix('').name
        imgprs = MedicalImageProcessor(image_path)
        imgprs.plot_single_slice(savefig=str(tmp_dir / (image_name+'.png')))
        content.append(Graphs.draw_little_title(image_name))
        base_info_df = imgprs.get_base_info()
        base_info_df['Value'] = base_info_df['Value'].astype('str')
        # Image base info table
        base_info_table = tuple([base_info_df.columns.to_list()] + base_info_df.values.tolist())
        content.append(Graphs.draw_table(*base_info_table, col_width=[120, 360]))
        content.append(Spacer(1, 20))
        content.extend(Graphs.draw_img(str(tmp_dir / (image_name+'.png')), width=7))
        content.append(Spacer(1, 20))

    # 生成pdf文件
    doc = SimpleDocTemplate(str(saved_dir / args.report_pdf), pagesize=letter)
    doc.build(content)

def report_single_image(args):
    image_path = Path(args.image_path)
    if not image_path.exists():
        print('The path of image is not exist, please check it!')
        sys.exit()
    saved_dir = Path(args.report_saved_dir)
    tmp_dir = Path('./tmp')

    # ----------------------report information generate----------------------------
    imgprs = MedicalImageProcessor(image_path)
    # Produce base information
    base_info_df = imgprs.get_base_info()
    base_info_df['Value'] = base_info_df['Value'].astype('str')
    meta_info_df = imgprs.metadata_info()

    # Produce figure
    imgprs.plot_single_slice(savefig=str(tmp_dir / 'single_img.png'))
    imgprs.plot_multi_slices(num=20, savefig=str(tmp_dir / 'multi_img.png'))

    # ----------------------pdf report generate----------------------------
    # 创建内容对应的空列表
    content = list()
    content.append(Graphs.draw_title(args.report_title))
    if args.report_describe:
        content.append(Graphs.draw_text(args.report_describe))

    # 添加段落文字
    content.append(Graphs.draw_text('The example data file path:', color='green'))
    content.append(Graphs.draw_text(args.image_path))

    # Image base info table
    content.append(Graphs.draw_little_title('Base information of the dicom series'))
    base_info_table = tuple([base_info_df.columns.to_list()] + base_info_df.values.tolist())
    content.append(Graphs.draw_table(*base_info_table, col_width=[120, 360]))

    # 添加图片
    content.append(Spacer(1, 20))
    content.append(Graphs.draw_text('Single image in middle one:', color='green'))
    content.extend(Graphs.draw_img(str(tmp_dir / 'single_img.png'), width=8))

    # 添加图片
    content.append(Spacer(1, 20))
    content.append(Graphs.draw_text('All slices:', color='green'))
    content.extend(Graphs.draw_img(str(tmp_dir / 'multi_img.png'), width=16))

    # Metadata base info table
    content.append(Graphs.draw_little_title('Metadata of the dicom series'))
    meta_info_table = tuple([meta_info_df.columns.to_list()] + meta_info_df.values.tolist())
    content.append(Graphs.draw_table(*meta_info_table, col_width=[120, 360]))

    # 生成pdf文件
    doc = SimpleDocTemplate(str(saved_dir / args.report_pdf), pagesize=letter)
    doc.build(content)


def main(args):
    print(args)

    # mkdir a template directory
    tmp_dir = Path('./tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.image_type == 'single':
        report_single_image(args)
    else:
        report_multi_image(args)

    # Delete the template files in the process
    if not args.report_tmp_keep:
        shutil.rmtree(str(tmp_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image information preview', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.report_saved_dir:
        Path(args.report_saved_dir).mkdir(parents=True, exist_ok=True)
    main(args)
