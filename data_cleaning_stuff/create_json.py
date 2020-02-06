import os
import json

base_path = r'D:\Masi_data\SPIE_2020\UploadData'
base_path = os.path.normpath(base_path)
dir_list = os.listdir(base_path)

json_f_name = r'data_list_frac_vol.json'
json_path = os.path.join(base_path, json_f_name)

# File names are hard coded
sh_f_name = r'dwi1K_sh.nii.gz'
fodf_f_name = r'fodf_sh.nii.gz'
mask_name = r'nodif_brain_mask.nii.gz'
volfrac_name = r'tissueVolFrac.nii'

json_dump = []

for each in dir_list:
    dir_path = os.path.join(base_path, each)

    sh_f_path = os.path.join(dir_path, sh_f_name)
    fodf_f_path = os.path.join(dir_path, fodf_f_name)
    mask_path = os.path.join(dir_path, mask_name)
    volfrac_path = os.path.join(dir_path, volfrac_name)

    data_dict = {'input_image': sh_f_path, 'output_image': fodf_f_path, 'mask': mask_path, 'vol_frac':volfrac_path}
    json_dump.append(data_dict)


new_data_dict = {}
new_data_dict['train'] = [json_dump[0]]
new_data_dict['validation'] = [json_dump[1]]
new_data_dict['test'] = [json_dump[2]]

with open(json_path, 'w') as json_file:
    json.dump(new_data_dict, json_file)
json_file.close()