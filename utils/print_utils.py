import argparse
import os
import sys
import torch
from utils import path_utils
import pandas as pd
import numpy as np
from argparse import Namespace
import ast
import pickle


def write_line(f, line):
    f.write(line + '\n')


def write_list(f, title, line_list):
    write_line(f, title)
    for line in line_list:
        write_line(f, "\t" + line)


def write_dict(f, title, dict_obj):
    write_line(f, title)
    for key, val in dict_obj.items():
        write_line(f, "\t" + f"{key} = {val}")


def print_exp_details(args):
    with open(path_utils.get_exp_details_path(args), 'w') as exp_details_file:
        write_line(exp_details_file, "# # # # # # # # # # # # # # # # # # # # # # # # # # # ")
        write_line(exp_details_file, f"Working on exp: {args.run_name}, version: {args.version}")
        write_line(exp_details_file, f"Current working directory: {os.getcwd()}")
        write_line(exp_details_file, f"")

        write_list(exp_details_file, "Current python path is:", sys.path)
        write_line(exp_details_file, f"")

        write_dict(exp_details_file, "Arguments are", vars(args))
        write_line(exp_details_file, f"")

        write_line(exp_details_file, "# # # # # # # # # # # # # # # # # # # # # # # # # # # ")


def print_erros(args, err_dict, name):
    err_pd = dict_to_pandas(name, err_dict)
    print_path = path_utils.get_model_err_path(args)
    if os.path.isfile(print_path):
        err_pd = pd.concat((pd.read_excel(print_path, index_col=0), err_pd))
    err_pd.to_excel(print_path)


def dict_to_pandas(name, base_dict):
    vals = []
    for val in base_dict.values():
        if isinstance(val, torch.Tensor):
            val = val.item()
        vals.append(val)

    return pd.DataFrame(np.array(vals)[None], columns=base_dict.keys(), index=[name])


def load_exp_details(exp_name, version):
    args_dict = {}
    with open(path_utils.get_exp_details_path(Namespace(run_name=exp_name, version=version)), 'r') as exp_details_file:
        while not exp_details_file.readline().startswith("Arguments are"):
            continue

        arg_line = exp_details_file.readline().split(maxsplit=2)
        while len(arg_line) == 3 and arg_line[1] == "=":
            args_dict[arg_line[0]] = parse_value(arg_line[2])
            arg_line = exp_details_file.readline().split(maxsplit=2)

    return Namespace(**args_dict)


def parse_value(arg_str):
    arg_value = arg_str.strip()
    if arg_value in ["True", "False"]:
        arg_value = argparse_bool(arg_value)
    elif arg_value == "None":
        arg_value = None
    elif is_int(arg_value):
        arg_value = int(arg_value)
    elif is_float(arg_value):
        arg_value = float(arg_value)
    elif arg_value.startswith("[") and arg_value.strip().endswith("]"):
        arg_value = ast.literal_eval(arg_value)

    return arg_value


def is_int(val):
    if val.isdigit():
        return True
    elif val.startswith("-") and val[1:].isdigit():
        return True
    else:
        return False


def is_float(val):
    if val.replace(".", "", 1).isdigit():
        return True
    elif val.startswith("-") and val[1:].replace(".", "", 1).isdigit():
        return True
    elif is_scientific_notation(val):
        return True
    else:
        return False


def is_scientific_notation(val):
    sc_split = val.split("e", 1)
    if len(sc_split) == 2:
        sc_val, sc_power = val.split("e", 1)
        if is_float(sc_val) and is_int(sc_power):
            return True

    return False


def argparse_bool(bool_str):
    assert isinstance(bool_str, str)
    if bool_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif bool_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
