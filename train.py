import os
import time
import numpy as np
from typing import Callable
import paddle
import random
from paddle.io import DataLoader
from model import TimeSeriesTransformer
from common import EarlyStopping
from common import adjust_learning_rate
from common import Experiment
from prepare import prep_env
from wind_turbine_data import get_turbine_loc


def val(experiment, model, data_loader, criterion):
    
    """
    Desc:
        Validation function
    Args:
        experiment:
        model:
        data_loader:
        criterion:
    Returns:
        The validation loss
    """
    validation_loss = []
    args = experiment.get_args()
    src_mask = TimeSeriesTransformer.generate_square_subsequent_mask(
        dim1=args["dec_seq_len"],
        dim2=args["enc_seq_len"]
        )


    tgt_mask = TimeSeriesTransformer.generate_square_subsequent_mask(
        dim1=args["dec_seq_len"],
        dim2=args["dec_seq_len"]
        )
    for i, (batch_x, batch_y) in enumerate(data_loader):
        sample, true = experiment.process_one_batch(model, batch_x, batch_y, src_mask, tgt_mask)
        loss = criterion(sample, true)
        validation_loss.append(loss.item())
    validation_loss = np.average(validation_loss)
    return validation_loss


def train_and_val(experiment, model, model_folder, is_debug=False):
    
    """
    Desc:
        Training and validation
    Args:
        experiment:
        model:
        model_folder: folder name of the model
        is_debug:
    Returns:
        None
    """
    args = experiment.get_args()
    train_data, train_loader = experiment.get_data(flag='train')
    val_data, val_loader = experiment.get_data(flag='val')

    path_to_model = os.path.join(args["checkpoints"], model_folder)
    

    early_stopping = EarlyStopping(patience=args["patience"], verbose=True)
    optimizer = experiment.get_optimizer(model)
    criterion = Experiment.get_criterion()

    
    src_mask = TimeSeriesTransformer.generate_square_subsequent_mask(
        dim1=args["dec_seq_len"],
        dim2=args["enc_seq_len"]
        )

    tgt_mask = TimeSeriesTransformer.generate_square_subsequent_mask( 
        dim1=args["dec_seq_len"],
        dim2=args["dec_seq_len"]
        )

    epoch_start_time = time.time()
    for epoch in range(1):
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            sample, truth = experiment.process_one_batch(model, batch_x, batch_y, src_mask, tgt_mask)
            loss = criterion(sample, truth)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.minimize(loss)
            optimizer.step()
        val_loss = val(experiment, model, val_loader, criterion)
        if is_debug:
            train_loss = np.average(train_loss)
            epoch_end_time = time.time()
            print("Epoch: {}, \nTrain Loss: {}, \nValidation Loss: {}".format(epoch, train_loss, val_loss))
            print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time

        # Early Stopping if needed
        early_stopping(val_loss, model, path_to_model, args["turbine_id"])
        if early_stopping.early_stop:
            print("Early stopped! ")
            break
        #adjust_learning_rate(optimizer, epoch + 1, args)


if __name__ == "__main__":
    fix_seed = 3407
    random.seed(fix_seed)
    paddle.seed(fix_seed)
    np.random.seed(fix_seed)

    settings = prep_env()
    #
    # Set up the initial environment
    # Current settings for the model
    cur_setup = 't{}_i{}_o{}_del{}_train{}_val{}'.format(
        settings["task"], settings["input_len"], settings["output_len"], settings["n_decoder_layers"],
        settings["train_size"], settings["val_size"]
    )
    start_train_time = time.time()
    end_train_time = start_train_time
    start_time = start_train_time

    tur_loc = get_turbine_loc(settings["loc_file_name"])
    path_to_model = os.path.join(settings["checkpoints"], cur_setup)
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)
    
    for epoch in range(settings["train_epochs"]):
        for tid in range(settings["capacity"]):
            settings["turbine_id"] = tid
            settings["tur_loc"] = tur_loc[tid]
            exp = Experiment(settings)
            print('\n>>>>>>> Training Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(tid))
            model = TimeSeriesTransformer(
                dim_val=settings["dim_val"],
                input_size=settings["input_size"], 
                enc_seq_len=settings["enc_seq_len"],
                dec_seq_len=settings["dec_seq_len"],
                max_seq_len=settings["max_seq_len"],
                out_seq_len=settings["output_len"], 
                n_decoder_layers=settings["n_decoder_layers"],
                n_encoder_layers=settings["n_encoder_layers"],
                n_heads=settings["n_heads"])
            if epoch !=0 or tid!=0:
                model.set_state_dict(paddle.load(path_to_model+'/'+'model'))
            train_and_val(exp, model=model, model_folder=cur_setup, is_debug=settings["is_debug"])
            paddle.device.cuda.empty_cache()
            if settings["is_debug"]:
                end_time = time.time()
                print("\nTraining the {}-th turbine in {} secs".format(tid, end_time - start_time))
                start_time = end_time
                end_train_time = end_time
    if settings["is_debug"]:
        print("\nTotal time in training {} turbines is "
              "{} secs".format(settings["capacity"], end_train_time - start_train_time))
