import os
import time
import numpy as np
import paddle
import pandas as pd
from model import TimeSeriesTransformer
from common import Experiment
from wind_turbine_data import WindTurbineData
from wind_turbine_data import preprocess
from test_data import TestData


def forecast_one(experiment, test_turbines, train_data, tur_loc):
    # type: (Experiment, TestData, WindTurbineData, tur_loc) -> np.ndarray
    """
    Desc:
        Forecasting the power of one turbine
    Args:
        experiment:
        test_turbines:
        train_data:
    Returns:
        Prediction for one turbine
    """
    args = experiment.get_args()
    tid = args["turbine_id"]
    model = TimeSeriesTransformer(
                dim_val=args["dim_val"],
                input_size=args["input_size"], 
                enc_seq_len=args["enc_seq_len"],
                dec_seq_len=args["dec_seq_len"],
                max_seq_len=args["max_seq_len"],
                out_seq_len=args["output_len"], 
                n_decoder_layers=args["n_decoder_layers"],
                n_encoder_layers=args["n_encoder_layers"],
                n_heads=args["n_heads"])
    model_dir = 't{}_i{}_o{}_del{}_train{}_val{}'.format(
        args["task"], args["input_len"], args["output_len"], args["n_decoder_layers"],
        args["train_size"], args["val_size"]
    )
    path_to_model = os.path.join(args["checkpoints"], model_dir, "model")
    model.set_state_dict(paddle.load(path_to_model))

    src_mask = TimeSeriesTransformer.generate_square_subsequent_mask(
        dim1=args["dec_seq_len"],
        dim2=args["enc_seq_len"]
        )


    tgt_mask = TimeSeriesTransformer.generate_square_subsequent_mask(
        dim1=args["dec_seq_len"],
        dim2=args["dec_seq_len"]
        )

    test_x, df_test_x = test_turbines.get_turbine(tid)
    df_test_x.insert(3, value=tur_loc[tid][1], column="loc_x")
    df_test_x.insert(4, value=tur_loc[tid][2], column="loc_y")
    cols = df_test_x.columns[args["start_col"]:]
    df_data = df_test_x[cols]

    cols = df_data.columns[2:]
    preprocess(df_data)
    scaler = train_data.get_scaler(tid)

    last_observ = df_data[-args["input_len"]:]
    last_observ[cols] = scaler.transform(last_observ[cols].values)
    seq_x = paddle.to_tensor(last_observ.values)
    sample_x = paddle.reshape(seq_x, [-1, seq_x.shape[-2], seq_x.shape[-1]])
    prediction = experiment.inference_one_sample(model, sample_x, src_mask, tgt_mask)
    prediction = scaler.inverse_transform(prediction[0])
    result = prediction.numpy()
    
    nan_prediction = pd.isna(prediction).any(axis=1)
    if nan_prediction.any():
        pred_mean = paddle.nanmean(prediction)
        df = pd.Dataframe(prediction)
        df.fillna(value=pred_mean, inplace=True)
        df.fillna(value=0, inplace=True)
        result = df.to_numpy()
    
    return result


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions
    """
    start_time = time.time()
    predictions = []
    settings["turbine_id"] = 0
    exp = Experiment(settings)
    # train_data = Experiment.train_data
    train_data = exp.load_train_data()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    locfile_path = os.path.join(dir_path, "normal_loc.csv")
    tur_loc = pd.read_csv(locfile_path).values
    if settings["is_debug"]:
        end_train_data_get_time = time.time()
        print("Load train data in {} secs".format(end_train_data_get_time - start_time))
        start_time = end_train_data_get_time
    test_x = Experiment.get_test_x(settings)
    if settings["is_debug"]:
        end_test_x_get_time = time.time()
        print("Get test x in {} secs".format(end_test_x_get_time - start_time))
        start_time = end_test_x_get_time
    for i in range(settings["capacity"]):
        settings["turbine_id"] = i
        # print('\n>>>>>>> Testing Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(i))
        prediction = forecast_one(exp, test_x, train_data, tur_loc)
        paddle.device.cuda.empty_cache()
        predictions.append(prediction)
        if settings["is_debug"] and (i + 1) % 10 == 0:
            end_time = time.time()
            print("\nElapsed time for predicting 10 turbines is {} secs".format(end_time - start_time))
            start_time = end_time
    return np.array(predictions)
