import numpy as np
import nibabel as nib
import random
import time


def load_nifty(path_to_file, data_type):
    start_time = time.time()
    nifti_data = nib.load(path_to_file)
    nifti_img = nifti_data.get_fdata(dtype=data_type)
    #nifti_data.uncache()
    end_time = time.time()
    print('\n Time Take to Read {}'.format(end_time - start_time))
    return nifti_img


def nifti_image_generator(inputPath, bs):
    # open the CSV file for reading
    #f = open(inputPath, "r")
    n_retrievals = 250000
    n_classes = 45
    # loop indefinitely
    while True:

        for vol_index, each_vol in enumerate(inputPath):

            # Load Nifti Volumes of Input, Output and Mask
            input_vol = load_nifty(each_vol['input_image'], data_type='float32')
            output_vol = load_nifty(each_vol['output_image'], data_type='float32')
            mask_vol = load_nifty(each_vol['mask'], data_type='float32')
            #frac_vol = load_nifty(each_vol['vol_frac'])
            # Convert mask_vol to int to save space
            mask_vol = np.int16(mask_vol)

            # Extract Voxel Indices
            true_vox_inds = np.where(mask_vol == 1)
            true_vox_inds = np.asarray(true_vox_inds)
            true_vox_inds = np.transpose(true_vox_inds)
            len_true_vox = len(true_vox_inds)

            current_retrieval = 0
            while current_retrieval < n_retrievals:
                # initialize our batches of images and labels
                images = np.empty((bs, n_classes))
                labels = np.empty((bs, n_classes))

                # X = np.empty((self.batch_size, self.n_classes))
                # y = np.empty((self.batch_size, self.n_classes))

                # Generate Random Inds for usage
                rand_inds = random.sample(range(len_true_vox - 1), bs)

                # Generate data
                for each_ind, each in enumerate(rand_inds):
                    # Retrieve indices
                    vox_inds = true_vox_inds[each, :]
                    images[each_ind, :] = np.squeeze(input_vol[vox_inds[0], vox_inds[1], vox_inds[2], :])
                    labels[each_ind, :] = np.squeeze(output_vol[vox_inds[0], vox_inds[1], vox_inds[2], :])

                current_retrieval = current_retrieval + bs
                yield (images, labels)

def nifti_smt_generator_simple(inputPath, bs):
    # open the CSV file for reading
    #f = open(inputPath, "r")
    n_retrievals = 100000
    n_classes = 3
    frac_vol_classes = 3
    # loop indefinitely
    while True:

        for vol_index, each_vol in enumerate(inputPath):

            # Load Nifti Volumes of Input, Output and Mask
            input_vol = load_nifty(each_vol['input_image_rdnn'], data_type='float32')
            mask_vol = load_nifty(each_vol['mask'], data_type='float32')
            diff_vol = load_nifty(each_vol['output_image_diff'], data_type='float32')
            trans_vol = load_nifty(each_vol['output_image_trans'], data_type='float32')
            intra_vol = load_nifty(each_vol['output_image_intra'], data_type='float32')

            # Convert mask_vol to int to save space
            mask_vol = np.int16(mask_vol)

            # Extract Voxel Indices
            true_vox_inds = np.where(mask_vol == 1)
            true_vox_inds = np.asarray(true_vox_inds)
            true_vox_inds = np.transpose(true_vox_inds)
            len_true_vox = len(true_vox_inds)

            print('\n New Volume Read {}'.format(each_vol['input_image']))

            current_retrieval = 0
            while current_retrieval < n_retrievals:
                # initialize our batches of images and labels
                images = np.empty((bs, n_classes))
                labels = np.empty((bs, frac_vol_classes))

                # X = np.empty((self.batch_size, self.n_classes))
                # y = np.empty((self.batch_size, self.n_classes))

                # Generate Random Inds for usage
                rand_inds = random.sample(range(len_true_vox - 1), bs)

                # Generate data
                for each_ind, each in enumerate(rand_inds):
                    # Retrieve indices
                    vox_inds = true_vox_inds[each, :]

                    images[each_ind, :] = np.squeeze(input_vol[vox_inds[0], vox_inds[1], vox_inds[2], 45:48])

                    labels[each_ind, 0] = np.squeeze(diff_vol[vox_inds[0], vox_inds[1], vox_inds[2]])
                    labels[each_ind, 1] = np.squeeze(trans_vol[vox_inds[0], vox_inds[1], vox_inds[2]])
                    labels[each_ind, 2] = np.squeeze(intra_vol[vox_inds[0], vox_inds[1], vox_inds[2]])

                    #labels[each_ind, 0:45] = np.squeeze(output_vol[vox_inds[0], vox_inds[1], vox_inds[2], :])

                    # The last three values in the labels correspond
                    # to the fractional volumes estimated per voxel
                    #labels[each_ind, 45:48] = np.squeeze(frac_vol[vox_inds[0], vox_inds[1], vox_inds[2], :])

                current_retrieval = current_retrieval + bs
                yield (images, labels)
