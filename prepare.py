import paddle


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    
    settings = {
        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": "./data/data138339/",
        "loc_file_name":"./data/data138339/sdwpf_baidukddcup2022_turb_location.CSV",
        "filename": "sdwpf_baidukddcup2022_full.CSV",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "work",
        "input_len": 144 * 6,
        "output_len": 288,
        "start_col": 3,
        "dim_val" : 256, 
        "n_heads" : 4, 
        "n_decoder_layers" : 2, 
        "n_encoder_layers" : 2, 
        "input_size" : 12, 
        "dec_seq_len" : 144 * 3, 
        "enc_seq_len" : 144 * 4, 
        "max_seq_len" : 144 * 4,
        "day_len": 144,
        "train_size": 150,
        "val_size": 15,
        "total_size": 245,
        "num_workers": 5,
        "train_epochs": 10,
        "batch_size": 32,
        "patience": 3,
        "lr": 1e-4,
        "lr_adjust": "type1",
        "gpu": 0,
        "capacity": 134,
        "tur_loc": [0., 0., 0.],
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        "is_debug": True
    }
    ###
    # Prepare the GPUs
    if paddle.device.is_compiled_with_cuda():
        settings["use_gpu"] = True
        paddle.device.set_device('gpu:{}'.format(settings["gpu"]))
    else:
        settings["use_gpu"] = False
        paddle.device.set_device('cpu')

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
