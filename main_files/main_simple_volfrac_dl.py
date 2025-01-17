import os
import numpy as np
import random
import argparse
import time
import json
import tensorflow as tf

from keras.callbacks import TensorBoard, ModelCheckpoint
from data_gens.data_generator_v2 import nifti_image_volfrac_generator
from data_gens.test_gen_v2 import test_fracvol_predictor_with_acc_v2
from models.sequential_models import build_nn_resnet_fracvol

# Critical, for Deep learning determinism
seed_value = 123
#tf.set_random_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_list', '-d', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\Final_Data\data_list_final.json',
                        help='Data List file for training stored in a JSON format')

    parser.add_argument('--model_dir', '-m', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\trained_resdnn_volfrac_model_all',
                        help='model output directory')

    parser.add_argument('--script_mode', required=False, type=str,
                        default=r'test',
                        help='Can run in two modes "train" or "test". If test then prior weight paths are needed')

    parser.add_argument('--prior_weights', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\trained_resdnn_volfrac_model_all\weights-improvement-17-0.00.hdf5',
                        help='Weights path for running the script in test mode')

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
    ad_data = all_data["additional"]

    print('Debug here')

    # Build the Data Sources from the Json file
    # and send to the data generator for construction

    # Build Model
    dl_model = build_nn_resnet_fracvol()

    if len(args.prior_weights) > 5:
        dl_model.load_weights(args.prior_weights)

    if args.script_mode == "train":

        # Generators
        trainGen = nifti_image_volfrac_generator(tr_data, bs=1000)
        validGen = nifti_image_volfrac_generator(val_data, bs=1000)

        # Callbacks for Tensorboard, Saving model with check-pointing
        tensor_board = TensorBoard(log_dir=model_base_path)

        model_ckpt_name = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        model_ckpt_path = os.path.join(model_base_path, model_ckpt_name)
        checkpoint = ModelCheckpoint(model_ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        callbacks_list = [checkpoint, tensor_board]

        # Fit the DataGenerators
        dl_model.fit_generator(generator=trainGen,
                               validation_data=validGen,
                               steps_per_epoch=250,
                               validation_steps=100,
                               epochs=40,
                               verbose=1,
                               callbacks=[tensor_board, checkpoint])

    elif args.script_mode == "test":

        # Test Volumes
        test_fracvol_predictor_with_acc_v2(dl_model=dl_model, test_data=ad_data, save_path=model_base_path)

if __name__ == '__main__':
    main()









