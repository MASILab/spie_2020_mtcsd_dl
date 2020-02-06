import os
import csv
import numpy as np

def main():

    base_path = r'D:\Masi_data\SPIE_2020\trained_patch_smt_model_w_est'
    base_path = os.path.normpath(base_path)
    base_dir_list = os.listdir(base_path)
    csv_f_name = 'csv_tr_val.csv'
    min_val_loss = 1000.0

    for each_w_dir in base_dir_list:

        # Construct CSV Path
        each_w_csv_path = os.path.join(base_path, each_w_dir, csv_f_name)

        with open(each_w_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    if len(row) != 0:
                        if float(row[5]) < min_val_loss:
                            min_val_loss = float(row[5])
                            print('Min Validation Loss: {} Detected at Epoch: {} for Weight Directory: {}'
                                  .format(row[5], row[0], each_w_dir))

                    line_count += 1
        csv_file.close()
    return None

if __name__=="__main__":
    main()