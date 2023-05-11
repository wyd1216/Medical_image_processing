import pprint
import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wprint(info, style=0):
    if isinstance(info, pathlib.PosixPath):
        info = str(info)

    # Determine the total length of the print line based on length of `info`
    if isinstance(info, str):
        tot_len = max(len(info) + 6, 80) if len(info) < 77 else max(len(info) + 6, 100)

        # Print top line of `=`
        if style <= 1:
            print('=' * tot_len)
        else:
            print('-' * tot_len)

        # Calculate the amount of spaces to be added to center the `info`
        space_len = (tot_len - len(info) - 6) // 2


        # Print middle line with `info` centered
        print('=', ' ' * space_len, f'\033[1;97;95m{info}\033[0m',
                  ' ' * (tot_len - space_len - 6 - len(info)), '=')
    else:
        tot_len = 100

        # Print top line of `=`
        if style <= 1:
            print('=' * tot_len)
        else:
            print('-' * tot_len)

        # Use `pprint` to print `info` if `style` is not 0
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(f'\033[1;97;95m{info}\033[0m')

    # Print bottom line of `=`
    if style <= 1:
        print('=' * tot_len)
    else:
        print('-' * tot_len)

def save_file_path_to_pandas_table(directory, save_path):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii.gz"):
                file_paths.append(os.path.abspath(os.path.join(root, file)))

    if len(file_paths) == 0:
        print("No .nii.gz files found in the directory")
        return

    file_path_table = pd.DataFrame({"file_path": file_paths})
    file_path_table.to_csv(save_path, index=False)
    wprint(f"File paths saved to {save_path}", style=1)


# plt func
def multi_image_show(array, cols=None):
    # Set up the plot
    subfig_num = len(array)
    if cols:
        cols = cols
    elif len(array) < 4:
        cols = subfig_num
    else:
        cols = 4
    rows, fig, ax = subplots_extension(subfig_num, cols)
    for i in range(subfig_num):
        _ = ax[i // cols, i % cols].imshow(array[i], 'gray')
        _ = ax[i // cols, i % cols].axis('off')  # 不显示坐标尺寸
        _ = ax[i // cols, i % cols].set_xticks([])
        _ = ax[i // cols, i % cols].set_yticks([])
    subplots_remove_residual(ax, subfig_num, rows, cols)
    # Save the plot to file (if applicable) and return the plot axis object
    return fig, ax


def subplots_remove_residual(ax, subfig_num, rows, cols):
    for i in range(subfig_num, cols * rows):
        ax[i // cols, i % cols].set_xticks([])
        ax[i // cols, i % cols].set_yticks([])
        ax[i // cols, i % cols].spines['left'].set_linewidth(0)
        ax[i // cols, i % cols].spines['right'].set_linewidth(0)
        ax[i // cols, i % cols].spines['top'].set_linewidth(0)
        ax[i // cols, i % cols].spines['bottom'].set_linewidth(0)


def subplots_extension(subfig_num, cols):
    rows = subfig_num // cols
    if subfig_num % cols > 0:
        rows += 1
    fig, ax = plt.subplots(
        rows, cols, sharex='col', sharey='row',
        figsize=(cols * 3,
                 rows * 3))  # 通过"sharex='col', sharey='row'"参数设置共享行列坐标轴
    if rows == 1:
        ax = np.array([ax])
    if cols == 1:
        ax = np.array([[x] for x in ax])
    return rows, fig, ax

