import os
import json

base_path = r'D:\Masi_data\SPIE_2020\Final_Data'
base_path = os.path.normpath(base_path)
dir_list = os.listdir(base_path)

json_f_name = r'data_list_init_pcnn_pred_smt.json'
json_path = os.path.join(base_path, json_f_name)

# File names are hard coded
pcnn_fname = r'patchcnn_pred.nii'
volfrac_fname = r'tissueVolFrac.nii'
smt_trans_fname = r'smt_extratrans.nii'
smt_diff_fname = r'smt_diff.nii'
smt_intra_fname = r'smt_intra.nii'
mask_name = r'nodif_brain_mask.nii'

json_dump = []

for each_idx, each in enumerate(dir_list):

    if each_idx == 7 or each_idx == 8:
        dir_path = os.path.join(base_path, each)

        pcnn_f_path = os.path.join(dir_path, pcnn_fname)
        volfrac_f_path = os.path.join(dir_path, volfrac_fname)
        smt_trans_f_path = os.path.join(dir_path, smt_trans_fname)
        smt_diff_f_path = os.path.join(dir_path, smt_diff_fname)
        smt_intra_f_path = os.path.join(dir_path, smt_intra_fname)
        mask_path = os.path.join(dir_path, mask_name)

        data_dict = {'input_image': pcnn_f_path, 'output_image_diff': smt_diff_f_path, 'output_image_trans': smt_trans_f_path, 'output_image_intra': smt_intra_f_path, 'mask': mask_path, 'vol_frac': volfrac_f_path}
        json_dump.append(data_dict)


new_data_dict = {}
new_data_dict['train'] = [json_dump[0]]
new_data_dict['validation'] = [json_dump[1]]
new_data_dict['test'] = [json_dump[1]]

with open(json_path, 'w') as json_file:
    json.dump(new_data_dict, json_file)
json_file.close()