from pathlib import Path
from shutil import copyfile


models_dir = Path(__file__).parents[2]
models_path = 'model_zoo/model_zoo'
# print(model_dir / models_path)
py_files = (models_dir / models_path).rglob('*')
# print(list(py_files))


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


line_replacement1 = '        if type(kernel_size) == int and kernel_size > 1:\n' \
                   '    '
line_replacement2 = '        else:\n' \
                    '    '

line_replacement3 = '        if type(kwargs.get(kernel_size)) == int and kwargs.get(kernel_size) > 1:\n'\
                    '    '


def replace_line(file_name, line_num):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = lines[line_num].replace('nn.Conv2d', 'Conv2dRF')
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


def modify(file, basic_conv_flag, conv2d_line_index, kernel_size_var):
    print(str(file))
    if basic_conv_flag == 1:
        with open(str(file), "r") as in_file:
            buf = in_file.readlines()
        with open(str(file), "w") as out_file:
            for i, line in enumerate(buf):
                if i == conv2d_line_index:
                    if kernel_size_var == 1:
                        line = line.replace(line, line_replacement1 + line.replace('nn.Conv2d', 'Conv2dRF') +
                                            line_replacement2 + line)
                    elif kernel_size_var == 0:
                        line = line.replace(line, line_replacement3 + line.replace('nn.Conv2d', 'Conv2dRF') +
                                            line_replacement2 + line)
                out_file.write(line)
    else:
        with open(str(file), "r") as in_file:
            conv_indices = []
            for i, line in enumerate(in_file):
                if 'nn.Conv2d' in line:
                    if '_size=1,' in line.partition('kernel')[2] or '_size=(' in line.partition('kernel')[2]:
                        pass
                    else:
                        conv_indices.append(i)
        for i in conv_indices:
            replace_line(str(file), i)


def scan_and_prepend_py_files():
    for py in list(py_files):
        print(py)
        new_py = models_dir/models_path/(py.stem+'_rf.py')
        copyfile(str(py), str(new_py))
        line_prepender(str(new_py), 'from convrf.conv_rf import Conv2dRF')
        # copyfile(py, 'incep'+'_rf.py')
        basic_conv_exists = 0
        nn_conv2d_line_index = 0
        kernel_size_var = 0
        with open(str(new_py), 'r') as f:
            for i, line in enumerate(f):
                if line.find('class BasicConv2d') >= 0:
                    basic_conv_exists = 1
                if basic_conv_exists == 1:
                    if 'nn.Conv2d' in line:
                        if 'kernel_size' in line:
                            nn_conv2d_line_index = i
                            kernel_size_var = 1
                        break
            modify(new_py, basic_conv_exists, nn_conv2d_line_index, kernel_size_var)


scan_and_prepend_py_files()

