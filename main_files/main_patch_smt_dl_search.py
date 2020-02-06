import os
import numpy as np
import random
import argparse
import time
import json
import tensorflow as tf

from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from data_gens.data_generator_smt_patch import nifti_smt_generator_patch
from data_gens.test_gen_smt_patch import test_smt_patch_predictor
from models.smt_models import build_smt_patch_resnet_fracvol_west

#from models.patch_models import build_sh_patch_resnet_fracvol
#from data_gens.test_gen_v2 import test_fracvol_patch_predictor_with_acc_v2
#from data_gens.data_generator_patch import nifti_image_volfrac_generator_patch


# Critical, for Deep learning determinism
seed_value = 123
#tf.set_random_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_list', '-d', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\Final_Data\data_list_init_pcnn_pred_smt.json',
                        help='Data List file for training stored in a JSON format')

    parser.add_argument('--model_dir', '-m', required=False, type=str,
                        default=r'D:\Masi_data\SPIE_2020\trained_patch_smt_model_w_est',
                        help='model output directory')

    parser.add_argument('--script_mode', required=False, type=str,
                        default=r'train',
                        help='Can run in two modes "train" or "test". If test then prior weight paths are needed')

    parser.add_argument('--prior_weights', required=False, type=str,
                        default=r'',
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

    # Step over Weights in steps of 10
    weight_hyper = 100
    weight_step = 10
    for weight_range in range(100):

        # Build the Data Sources from the Json file
        # and send to the data generator for construction

        print('Building Model with the Weight for Loss set at: mn ,. {}'.format(weight_hyper))

        # Create Directory for operation with the suggested weight_hyper
        model_base_weight_name = 'model_w_' + str(weight_hyper)
        model_base_weight_path = os.path.join(model_base_path, model_base_weight_name)
        if os.path.exists(model_base_weight_path) == False:
            os.mkdir(model_base_weight_path)

        # Build Model
        dl_model = build_smt_patch_resnet_fracvol_west(weight_hyper)

        # Increment by the step size
        weight_hyper = weight_hyper + weight_step

        patch_crop = [3, 3, 3]

        # The condition below for checking existence of prior weights is a cheap hack, PLEASE IMPROVE
        if len(args.prior_weights) > 5:
            dl_model.load_weights(args.prior_weights)

        if args.script_mode == "train":

            # Generators
            trainGen = nifti_smt_generator_patch(tr_data, bs=1000, patch_size=patch_crop)
            validGen = nifti_smt_generator_patch(val_data, bs=1000, patch_size=patch_crop)

            model_ckpt_name = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
            model_ckpt_path = os.path.join(model_base_weight_path, model_ckpt_name)

            model_csv_path = os.path.join(model_base_weight_path, 'csv_tr_val.csv')

            # Callbacks for Tensorboard, Saving model with checkpointing
            tensor_board = TensorBoard(log_dir=model_base_weight_path)
            checkpoint = ModelCheckpoint(model_ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            csvlogger = CSVLogger(model_csv_path)

            callbacks_list = [checkpoint, tensor_board, csvlogger]

            # Fit the DataGenerators
            dl_model.fit_generator(generator=trainGen,
                                   validation_data=validGen,
                                   steps_per_epoch=200,
                                   validation_steps=200,
                                   epochs=50,
                                   verbose=1,
                                   callbacks=callbacks_list)

        elif args.script_mode == "test":
            # Test Volumes
            test_smt_patch_predictor(dl_model=dl_model, test_data=test_data, save_path=model_base_path)

if __name__ == '__main__':
    main()









