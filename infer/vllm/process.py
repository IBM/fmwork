import os
import csv
import sys
import re
import json
from datetime import datetime

servers = {
    'css-host-180': 'H100',
    'css-host-181': 'H100',
    'css-host-183': 'H100',
    'css-host-160': 'A100'
}

def traverse_directory(rdir):
    subdirs = []

    for root, dirs, files in os.walk(rdir):
        for dir_name in dirs:
            subdir_path = os.path.join(root, dir_name)
           
            subdir_path = subdir_path.replace(rdir, '').lstrip(os.sep)
            
            path_parts = subdir_path.split(os.sep)
            if len(path_parts) == 4:
                subdir_path = subdir_path.replace(os.sep, ',')
                subdirs.append(subdir_path)

    return subdirs

def find_fmwork_gen_lines(subdir, rdir):
    leaf_dir = os.path.join(rdir, subdir.replace(',', os.sep))
    exp_log_path = os.path.join(leaf_dir, 'exp.log')
    fmwork_gen_lines = []

    if os.path.isfile(exp_log_path):
        with open(exp_log_path, 'r') as file:
            for line in file:
                if line.startswith("FMWORK GEN"):
                    line = line.replace(' ', ',')
                    fmwork_gen_lines.append(line.strip())

    return fmwork_gen_lines

def find_vllmversion(subdir, rdir):
    leaf_dir = os.path.join(rdir, subdir.replace(',', os.sep))
    pip_list_path = os.path.join(leaf_dir, 'utils', 'pip-list')
    if os.path.exists(pip_list_path):
        with open(pip_list_path, 'r') as pip_list_file:
            pip_list_content = pip_list_file.read()

        vllm_line = re.search(r'^vllm\s+(.*)$', pip_list_content, re.MULTILINE)
        if vllm_line:
            vllm_parts = vllm_line.group(1).split()
            vllm_version=f"{'vllm=='}{vllm_parts[0]}"

    return vllm_version

def find_modelandprec(subdir, rdir):
    leaf_dir = os.path.join(rdir, subdir.replace(',', os.sep))
    exp_cmd_path = os.path.join(leaf_dir, 'exp.cmd')
    if os.path.exists(exp_cmd_path):
        with open(exp_cmd_path, 'r') as exp_cmd_file:
            exp_cmd_content = exp_cmd_file.read()

        model_precision = re.search(r'-m\s+(\S+)/(\S+)', exp_cmd_content)
        if model_precision:
            model, precision = model_precision.groups()

            if precision == "base":
                precision = "fp16"
        
            model_parts = model.rsplit('/', 1)
            model = model_parts[-1]

    return model,precision

def parse_extraparams(subdir, rdir):
    # Join the root directory and subdirectory
    leaf_dir = os.path.join(rdir, subdir.replace(',', os.sep))
    
    # Look one level above the joined path for params.json
    parent_dir = os.path.dirname(leaf_dir)
    params_path = os.path.join(parent_dir, 'params.json')
    
    # Read the params.json file
    with open(params_path, 'r') as file:
        params = json.load(file)
    
    # Get the value for the key 'extraparams'
    extraparams = params['extraparams'][0]

    env_path = os.path.join(leaf_dir, 'utils', 'env')
    
    vllm_vars = []
    with open(env_path, 'r') as file:
        for line in file:
            if line.startswith('VLLM'):
                vllm_vars.append(line.strip())
    
    # Join VLLM variables into a single string
    vllm_vars_str = ' '.join(vllm_vars)

    return extraparams,vllm_vars_str


def modify_subdir(subdir, fmwork_gen_line):
    subdir_parts = subdir.split(',')
    fmwork_gen_parts = fmwork_gen_line.split(',')

    fmwork_gen_parts = fmwork_gen_parts[2:]

    if len(subdir_parts) >= 4 and len(fmwork_gen_parts) > 0:
        subdir_parts[3] = fmwork_gen_parts[0]

    fmwork_gen_metrics = fmwork_gen_parts[-9:]

    modified_fmwork_gen_line = ','.join(fmwork_gen_metrics)

    date_str = subdir_parts[2][:8]
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    year = date_obj.year
    month = date_obj.month

    if 1 <= month <= 3:
        quarter = 1
    elif 4 <= month <= 6:
        quarter = 2
    elif 7 <= month <= 9:
        quarter = 3
    else:
        quarter = 4

    nQyear = f"{quarter}Q{year}"

    server_name = subdir_parts[1]
    gpu_type = servers.get(server_name, 'Unknown')

    hwc = fmwork_gen_parts[4] if len(fmwork_gen_parts) > 2 else ''

    modified_subdir = f"{nQyear},{','.join(subdir_parts)},{gpu_type},{hwc}"

    return modified_subdir, modified_fmwork_gen_line

def write_to_csv(subdirs, rdir, output_file='fmworkdata.csv'):
    data_parallelism=1
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['work','user','host','btim','etim','hw','hwc','back','mm','prec','dp','ii','oo','bb','tp','med','ttft','gen','itl','thp','extraparams','vllmvars']
        writer.writerow(header)
        for subdir in subdirs:
            fmwork_gen_lines = find_fmwork_gen_lines(subdir, rdir)
            vllm_version = find_vllmversion(subdir, rdir)
            model,prec = find_modelandprec(subdir, rdir)
            extraparams,vllmvars = parse_extraparams(subdir, rdir)
            for line in fmwork_gen_lines:
                modified_subdir, modified_fmwork_gen_line = modify_subdir(subdir, line)
                combined_list = modified_subdir.split(',') + [vllm_version, model, prec, data_parallelism] + modified_fmwork_gen_line.split(',') + [extraparams, vllmvars]
                writer.writerow(combined_list)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process.py <rdir_path>")
        sys.exit(1)

    rdir = sys.argv[1]
    subdirs = traverse_directory(rdir)
    write_to_csv(subdirs, rdir)
    print(f"FMWORK GEN data in all subdirectories have been written to fmworkdata.csv")
