import os
import numpy as np
import random
import argparse
import time
import json
import tensorflow as tf

from keras.callbacks import TensorBoard, ModelCheckpoint
from data_gens.data_generator_patch import nifti_image_generator_patch
from data_gens.test_gen_v2 import test_patch_predictor_v2
from models.patch_models import build_sh_patch_resnet

# Critical, for Deep learning determinism
seed_value = 123
#tf.set_random_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_list', '-d', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\UploadData\data_list_mod.json',
                        help='Data List file for training stored in a JSON format')

    parser.add_argument('--model_dir', '-m', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\trained_models_patch_v2',
                        help='model output directory')

    parser.add_argument('--script_mode', required=False, type=str,
                        default=r'test',
                        help='Can run in two modes "train" or "test". If test then prior weight paths are needed')

    parser.add_argument('--prior_weights', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\trained_models_patch_v2\weights-improvement-20-0.00.hdf5 ',
                        help='Weights path for running the script in test mode')

    args = parser.parse_args()

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

    print('Debug here')

    # Build the Data Sources from the Json file
    # and send to the data generator for construction
    params = {'batch_size': 1000,
              'n_classes': 28,
              'n_channels': 1,
              'shuffle': True,
              }

    # Build Model
    dl_model = build_sh_patch_resnet()

    patch_crop = [3, 3, 3]

    # The condition below for checking existence of prior weights is a cheap hack, PLEASE IMPROVE
    if len(args.prior_weights) > 5:
        dl_model.load_weights(args.prior_weights)

    if args.script_mode == "train":

        # Generators
        trainGen = nifti_image_generator_patch(tr_data, bs=1000, patch_size=patch_crop)
        validGen = nifti_image_generator_patch(val_data, bs=1000, patch_size=patch_crop)

        # Callbacks for Tensorboard, Saving model with checkpointing
        tensor_board = TensorBoard(log_dir=model_base_path,)

        model_ckpt_name = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        model_ckpt_path = os.path.join(model_base_path, model_ckpt_name)
        checkpoint = ModelCheckpoint(model_ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        callbacks_list = [checkpoint, tensor_board]

        # Fit the DataGenerators
        dl_model.fit_generator(generator=trainGen,
                               validation_data=validGen,
                               steps_per_epoch=100,
                               validation_steps=100,
                               epochs=20,
                               verbose=1,
                               callbacks=callbacks_list)

    elif args.script_mode == "test":

        # Test Volumes
        test_patch_predictor_v2(dl_model=dl_model, test_data=test_data, save_path=model_base_path)


if __name__ == '__main__':
    main()









