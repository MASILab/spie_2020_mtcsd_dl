import os
import numpy as np
import argparse
import json
from data_gens.test_gen_v2 import test_save_csd_acc_v2
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_list', '-d', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\Final_Data\data_list_final_csd.json',
                        help='Data List file for training stored in a JSON format')

    parser.add_argument('--model_dir', '-m', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\scsd_acc_model_all',
                        help='model output directory')

    args = parser.parse_args()

    # Print out the arguments passed in
    for arg in vars(args):
        print('Argument Detected {}'.format(arg))
        print(getattr(args, arg))

    # Create the Model directory if non-existent
    model_base_path = args.model_dir
    model_base_path = os.path.normpath(model_base_path)
    if os.path.exists(model_base_path) is False:
        os.mkdir(model_base_path)


    # Load Json file
    data_list_path = args.data_list
    data_list_path = os.path.normpath(data_list_path)
    all_data = json.load(open(data_list_path))
    tr_data = all_data["train"]
    val_data = all_data["validation"]
    test_data = all_data["test"]

    test_save_csd_acc_v2(test_data=test_data, save_path=model_base_path)

if __name__ == '__main__':
    main()