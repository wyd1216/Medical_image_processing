import tempfile
import os
from pathlib import Path
from reportlab.pdfbase import pdfmetrics  # 注册字体
from reportlab.pdfbase.ttfonts import TTFont  # 字体类
from reportlab.platypus import Table, SimpleDocTemplate, Paragraph, Image, PageBreak  # 报告内容相关类
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # 文本样式
from reportlab.lib import colors  # 颜色模块
from reportlab.graphics.charts.barcharts import VerticalBarChart  # 图表类
from reportlab.graphics.charts.legends import Legend  # 图例类
from reportlab.graphics.shapes import Drawing  # 绘图工具
from reportlab.lib.pagesizes import letter  # 页面的标志尺寸(8.5*inch, 11*inch)
from reportlab.lib.units import cm  # 单位：cm
from PIL import Image as PILImage

pdfmetrics.registerFont(TTFont('SIMSUN', 'SIMSUN.TTC'))


# 封装不同内容对应的函数
# 创建一个Graphs类，通过不同的静态方法提供不同的报告内容，包括：标题、普通段落、图片、表格和图表。函数中的相关数据目前绝大多数都是固定值，可以根据情况自行设置成相关参数。
# Graphs类的全部代码，请+v：CoderWanFeng

class Graphs:
    # 绘制标题
    @staticmethod
    def draw_title(title: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 拿到标题样式
        ct = style['Heading1']
        # 单独设置样式相关属性
        ct.fontName = 'SimSun'  # 字体名
        ct.fontSize = 18  # 字体大小
        ct.leading = 50  # 行间距
        ct.textColor = colors.black  # 字体颜色
        ct.alignment = 1  # 居中
        ct.bold = True
        # ct.fontSize =  # 字体大小
        # ct.leading =  # 行间距
        # ct.textColor = colors.green  # 字体颜色
        # ct.alignment =  # 居中
        # ct.bold = True
        # 创建标题对应的段落，并且返回
        return Paragraph(title, ct)

    # 绘制小标题
    @staticmethod
    def draw_little_title(title: str, heading: str = 'Heading2'):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 拿到标题样式
        ct = style[heading]
        # 单独设置样式相关属性
        ct.fontName = 'SimSun'  # 字体名
        ct.fontSize = 14  # 字体大小
        ct.leading = 50  # 行间距
        ct.textColor = colors.blue  # 字体颜色
        ct.firstLineIndent = 0  # 第一行开头空格
        # 创建标题对应的段落，并且返回
        return Paragraph(title, ct)

    # 绘制普通段落内容
    @staticmethod
    def draw_text(text: str, color: str = 'black'):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 获取普通样式
        ct = style['Normal']
        ct.fontName = 'SimSun'
        ct.fontSize = 12
        ct.wordWrap = 'CJK'  # 设置自动换行
        ct.alignment = 0  # 左对齐
        ct.firstLineIndent = 0  # 第一行开头空格
        #
        ct.textColor = getattr(colors, color)
        # ct.textColor = colors.black # 字体颜色
        ct.leading = 25
        return Paragraph(text, ct)

    # 绘制表格
    @staticmethod
    def draw_table(*args, col_width=120):
        # Define a paragraph style for the table cells
        cell_style = ParagraphStyle(
            name='cell_style',
            # fontSize=12,
            textColor=colors.black,
            alignment=0,  # Left alignment
        )
        # Convert the table data to include Paragraph instances
        args = [[Paragraph(cell, cell_style) for cell in row] for row in args]

        style = [
            ('FONTNAME', (0, 0), (-1, -1), 'SIMSUN'),  # 字体
            ('FONTSIZE', (0, 0), (-1, 0), 12),  # 第一行的字体大小
            ('FONTSIZE', (0, 1), (-1, -1), 10),  # 第二行到最后一行的字体大小
            ('BACKGROUND', (0, 0), (-1, 0), '#d5dae6'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 第一行水平居中
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),  # 第二行到最后一行左右左对齐
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.darkslategray),  # 设置表格内文字颜色
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # 设置表格框线为grey色，线宽为0.5
            # ('SPAN', (0, 1), (0, 2)),  # 合并第一列二三行
            # ('SPAN', (0, 3), (0, 4)),  # 合并第一列三四行
            # ('SPAN', (0, 5), (0, 6)),  # 合并第一列五六行
            # ('SPAN', (0, 7), (0, 8)),  # 合并第一列五六行
        ]
        table = Table(args, colWidths=col_width, style=style)
        return table

        # 创建图表

    @staticmethod
    def draw_bar(bar_data: list, ax: list, items: list):
        drawing = Drawing(500, 250)
        bc = VerticalBarChart()
        bc.x = 45  # 整个图表的x坐标
        bc.y = 45  # 整个图表的y坐标
        bc.height = 200  # 图表的高度
        bc.width = 350  # 图表的宽度
        bc.data = bar_data
        bc.strokeColor = colors.black  # 顶部和右边轴线的颜色
        bc.valueAxis.valueMin = 5000  # 设置y坐标的最小值
        bc.valueAxis.valueMax = 26000  # 设置y坐标的最大值
        bc.valueAxis.valueStep = 2000  # 设置y坐标的步长
        bc.categoryAxis.labels.dx = 2
        bc.categoryAxis.labels.dy = -8
        bc.categoryAxis.labels.angle = 20
        bc.categoryAxis.categoryNames = ax

        # 图示
        leg = Legend()
        leg.fontName = 'SIMSUN'
        leg.alignment = 'right'
        leg.boxAnchor = 'ne'
        leg.x = 475  # 图例的x坐标
        leg.y = 240
        leg.dxTextSpace = 10
        leg.columnMaximum = 3
        leg.colorNamePairs = items
        drawing.add(leg)
        drawing.add(bc)
        return drawing

    # # 绘制图片
    # @staticmethod
    # def draw_img(path, width=8, height=None):
    #     pil_image = PILImage.open(path)
    #     image_width, image_height = pil_image.size
    #     img = Image(path)  # 读取指定路径下的图片
    #     img.drawWidth = width * cm  # 设置图片的宽度
    #     if height:  # 设置图片的高度
    #         img.drawHeight = height * cm
    #     else:
    #         img.drawHeight = int(image_height * (width / image_width)) * cm
    #     return img

    @staticmethod
    def draw_img(path, width=8, height=None):
        pil_image = PILImage.open(path)
        image_width, image_height = pil_image.size
        img = Image(path)  # 读取指定路径下的图片
        max_height = 648
        if not height:  # 设置图片的高度
            height = int(image_height * (width / image_width))
        width = width * cm
        height = height * cm
        img.drawWidth = width  # 设置图片的宽度
        img.drawHeight = height
        if img.drawHeight > max_height:
            image_segments = []
            elements = []
            # elements.append(PageBreak())
            crop_height = int(max_height * (image_width / width))
            temp_dir = tempfile.mkdtemp(dir='./tmp')
            for y in range(0, int(image_height), int(crop_height)):
                cropped_image = pil_image.crop((0, y, image_width, min(y + crop_height, image_height)))
                cropped_image_file = os.path.join(temp_dir, f"temp_cropped_image_{y}.png")
                print('tmp_path=', cropped_image_file)
                cropped_image.save(cropped_image_file)
                image_segments.append(cropped_image_file)
            # Add the image segments to the PDF
            for i, image_segment in enumerate(image_segments):
                if i == len(image_segments) - 1:
                    rl_image = Image(image_segment, width=width, height=height % max_height)
                else:
                    rl_image = Image(image_segment, width=width, height=max_height - 20)
                elements.append(rl_image)
                elements.append(PageBreak())
            return elements
        return [img]
