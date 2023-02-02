import pprint
import os
import pathlib
import pandas as pd

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
        pprint(f'\033[1;97;95m{info}\033[0m')

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


