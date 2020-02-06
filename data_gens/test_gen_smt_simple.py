import os
import nibabel as nib
import numpy as np
import time
from utils.metrics import calc_acc_numpy

def load_nifty(path_to_file, data_type):

    nifti_data = nib.load(path_to_file)
    nifti_img = nifti_data.get_fdata(dtype=data_type)
    nifti_data.uncache()
    return nifti_img

def save_nifti(predicted_vol, path_to_nifti_header, saver_path):

    nib_img = nib.Nifti1Image(predicted_vol, nib.load(path_to_nifti_header).affine, nib.load(path_to_nifti_header).header)
    # Grab ID from path to header
    f_name = path_to_nifti_header.split('\\')
    f_path = os.path.join(saver_path, f_name[-2] + '.nii.gz')
    nib.save(nib_img, f_path)

def save_nifti_acc(predicted_vol, path_to_nifti_header, saver_path):

    nib_img = nib.Nifti1Image(predicted_vol, nib.load(path_to_nifti_header).affine, nib.load(path_to_nifti_header).header)
    # Grab ID from path to header
    f_name = path_to_nifti_header.split('\\')
    f_path = os.path.join(saver_path, f_name[-2] + '_acc' + '.nii.gz')
    nib.save(nib_img, f_path)

def test_predictor(dl_model, test_data, save_path):

    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape
        pred_vol = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 45))
        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        ip_voxel = np.reshape(ip_voxel, [1, 45])
                        t_pred = dl_model.predict(ip_voxel)
                        pred_vol[x, y, z, :] = np.squeeze(t_pred)

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        save_nifti(pred_vol, each_vol['output_image'], vol_saver_path)

def test_predictor_v2(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape
        pred_vol = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 45))
        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        ip_voxel = np.reshape(ip_voxel, [1, 45])
                        t_pred = dl_model.predict(ip_voxel)
                        pred_vol[x, y, z, :] = np.squeeze(t_pred)

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        save_nifti(pred_vol, each_vol['output_image'], vol_saver_path)

def test_fracvol_predictor_v2(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape
        pred_vol = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 48))
        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        ip_voxel = np.reshape(ip_voxel, [1, 45])
                        t_pred = dl_model.predict(ip_voxel)
                        pred_vol[x, y, z, :] = np.squeeze(t_pred)

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        save_nifti(pred_vol, each_vol['output_image'], vol_saver_path)

def test_fracvol_predictor_with_acc_v2(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape
        pred_vol = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 48))
        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        ip_voxel = np.reshape(ip_voxel, [1, 45])
                        t_pred = dl_model.predict(ip_voxel)
                        pred_vol[x, y, z, :] = np.squeeze(t_pred)

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        save_nifti(pred_vol, each_vol['output_image'], vol_saver_path)

        print('Predicted Volume Saved ... \n ')
        #### Calculate ACC
        print('Calculating ACC')

        acc_vol = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2]))
        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        op_voxel = np.squeeze(output_vol[x, y, z, :])
                        pred_voxel = np.squeeze(pred_vol[x, y, z, :])
                        acc_vol[x, y, z] = calc_acc_numpy(op_voxel, pred_voxel)

        save_nifti_acc(acc_vol, each_vol['output_image'], vol_saver_path)

def test_smt_simple_predictor(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    batch_size = 1000
    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image_rdnn'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')
        diff_vol = load_nifty(each_vol['output_image_diff'], data_type='float32')
        trans_vol = load_nifty(each_vol['output_image_trans'], data_type='float32')
        intra_vol = load_nifty(each_vol['output_image_intra'], data_type='float32')

        #input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        #output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        #mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape
        pred_vol = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 3))

        batch_collector = np.zeros((batch_size, 3))
        batch_inds = np.zeros((batch_size, 3), dtype='int16')
        batch_counter = 0

        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        #ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        #ip_voxel = np.reshape(ip_voxel, [1, 45])

                        #sh_ip_voxel = np.squeeze(input_vol[x - 1:x + 2,
                        #                                   y - 1:y + 2,
                        #                                   z - 1:z + 2, 0]
                        #                         )

                        tissue_ip_voxel = np.squeeze(input_vol[x, y, z, 45:48])

                        batch_collector[batch_counter, :] = tissue_ip_voxel

                        batch_inds[batch_counter, :] = [x, y, z]
                        batch_counter = batch_counter + 1

                        #ip_voxel = np.reshape(ip_voxel, [1, 3, 3, 3, 45])
                        if batch_counter == batch_size:
                            # Make Preds
                            batch_preds = dl_model.predict(batch_collector)

                            # Assign Predictions to Prediction Volume
                            for per_pred in range(len(batch_collector)):
                                ret_x = batch_inds[per_pred, 0]
                                ret_y = batch_inds[per_pred, 1]
                                ret_z = batch_inds[per_pred, 2]

                                pred_vol[ret_x, ret_y, ret_z, :] = batch_preds[per_pred, :]

                            # Set counter to zero and reinitiate batch collector and batch inds to zeros
                            batch_collector = np.zeros((batch_size, 3))
                            batch_inds = np.zeros((batch_size, 3), dtype='int16')
                            batch_counter = 0

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        save_nifti(pred_vol, each_vol['output_image_diff'], vol_saver_path)

        print('Predicted Volume Saved ... \n ')

        '''
        #### Calculate ACC
        print('Calculating ACC')

        acc_vol = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2]))
        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        op_voxel = np.squeeze(output_vol[x, y, z, :])
                        pred_voxel = np.squeeze(pred_vol[x, y, z, :])
                        acc_vol[x, y, z] = calc_acc_numpy(op_voxel, pred_voxel)

        save_nifti_acc(acc_vol, each_vol['output_image'], vol_saver_path)
        '''

def test_patch_predictor_v2(dl_model, test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)

        vol_dims = mask_vol.shape
        pred_vol = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2], 45))
        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        #ip_voxel = np.squeeze(input_vol[x, y, z, :])
                        #ip_voxel = np.reshape(ip_voxel, [1, 45])
                        ip_voxel = np.squeeze(input_vol[x - 1:x + 2,
                                                        y - 1:y + 2,
                                                        z - 1:z + 2, :])
                        ip_voxel = np.reshape(ip_voxel, [1, 3, 3, 3, 45])
                        t_pred = dl_model.predict(ip_voxel)
                        pred_vol[x, y, z, :] = np.squeeze(t_pred)

        end_time = time.time()
        time_taken = end_time - start_time
        print('Predictions Completed for Vol {} & Time Taken was {} \n'.format(vol_index, time_taken))
        print('Saving predicted volume')

        save_nifti(pred_vol, each_vol['output_image'], vol_saver_path)



def test_save_csd_acc_v2(test_data, save_path):

    # An attempt to reduce inference time as the v1 takes over 15 minutes per volume...
    vol_saver_path = os.path.join(save_path, 'predicted_volumes')
    if os.path.exists(vol_saver_path) is False:
        os.mkdir(vol_saver_path)

    batch_size = 1000
    for vol_index, each_vol in enumerate(test_data):

        start_time = time.time()

        # Load Nifti Volumes of Input, Output and Mask
        #input_vol = load_nifty(each_vol['input_image'], data_type='float32')
        output_vol = load_nifty(each_vol['output_image'], data_type='float32')
        csd_vol = load_nifty(each_vol['csd_img'], data_type='float32')

        mask_vol = load_nifty(each_vol['mask'], data_type='float32')

        # Convert mask_vol to int to save space
        mask_vol = np.int16(mask_vol)
        vol_dims = mask_vol.shape

        #### Calculate ACC
        print('Calculating ACC')

        acc_vol = np.zeros((vol_dims[0], vol_dims[1], vol_dims[2]))
        for x in range(vol_dims[0]):
            print(x)
            for y in range(vol_dims[1]):
                for z in range(vol_dims[2]):
                    if mask_vol[x, y, z] == 1:
                        op_voxel = np.squeeze(output_vol[x, y, z, :])
                        pred_voxel = np.squeeze(csd_vol[x, y, z, :])
                        acc_vol[x, y, z] = calc_acc_numpy(op_voxel, pred_voxel)

        save_nifti_acc(acc_vol, each_vol['output_image'], vol_saver_path)



