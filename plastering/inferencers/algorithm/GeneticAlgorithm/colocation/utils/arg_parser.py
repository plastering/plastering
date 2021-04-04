"""Parsing arguments
"""
import argparse
import sys
from ..data_loader import config_loader

def my_parse_args(myargs) -> config_loader.ColocationConfig:
    '''
    parser = argparse.ArgumentParser(prog='colocation')
    parser.add_argument(
        '-g', '--gen-config', dest='gen_config', action='store_true')
    parser.add_argument('-c', '--config', dest='config_path', required=True)
    parser.add_argument('-m', '--corr-matrix-path', dest='corr_matrix_path')
    parser.add_argument('-o', '--output-path', dest='output_path')
    parser.add_argument('-j', '--job_name', dest='job_name')
    commands = parser.parse_args(myargs[1:])
    if commands.gen_config:
        config = config_loader.ColocationConfig()
        print(config)
        with open(commands.config_path, 'w') as file:
            file.write(config.to_json())
        exit(0)
    '''
    config = config_loader.load_config(myargs[3])

    config.corr_matrix_path = myargs[1]

    return config
'''
def my_parse_args(myargs) -> config_loader.ColocationConfig:
    parser = argparse.ArgumentParser(prog='colocation')
    parser.add_argument(
        '-g', '--gen-config', dest='gen_config', action='store_true')
    parser.add_argument('-c', '--config', dest='config_path', required=True)
    parser.add_argument('-m', '--corr-matrix-path', dest='corr_matrix_path')
    parser.add_argument('-o', '--output-path', dest='output_path')
    parser.add_argument('-j', '--job_name', dest='job_name')
    commands = parser.parse_args(myargs[1:])
    if commands.gen_config:
        config = config_loader.ColocationConfig()
        print(config)
        with open(commands.config_path, 'w') as file:
            file.write(config.to_json())
        exit(0)

    config = config_loader.load_config(commands.config_path)

    if commands.corr_matrix_path is not None:
        config.corr_matrix_path = commands.corr_matrix_path
    return config
'''
def parse_args() -> config_loader.ColocationConfig:
    """parse argument
    """
    parser = argparse.ArgumentParser(prog='colocation')
    parser.add_argument(
        '-g', '--gen-config', dest='gen_config', action='store_true')
    parser.add_argument('-c', '--config', dest='config_path', required=True)
    parser.add_argument('-m', '--corr-matrix-path', dest='corr_matrix_path')
    parser.add_argument('-o', '--output-path', dest='output_path')
    parser.add_argument('-j', '--job_name', dest='job_name')
    commands = parser.parse_args(sys.argv[1:])
    if commands.gen_config:
        config = config_loader.ColocationConfig()
        print(config)
        with open(commands.config_path, 'w') as file:
            file.write(config.to_json())
        exit(0)

    config = config_loader.load_config(commands.config_path)

    if commands.corr_matrix_path is not None:
        config.corr_matrix_path = commands.corr_matrix_path

    return config
