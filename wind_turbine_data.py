import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset


# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


class Scaler(object):
    """
    Desc: Normalization utilities
    """
    def __init__(self):
        self.mean = np.zeros((1))
        self.std = np.ones((1))

    def fit(self, data):
        # type: (paddle.tensor) -> None
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        for i in range(len(self.std)):
            if self.std[i] == 0: self.std[i] = 1
    def transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        
        
        for i in range(len(self.mean)):
            
            data[:,i] = (data[:,i] - self.mean[i]) / self.std[i]

        
        return data 

    def inverse_transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        i = len(self.mean) - 1
        data = data * self.std[i] + self.mean[i]
        
        return data


class WindTurbineData(Dataset):
    """
    Desc: Wind turbine power generation data
          Here, e.g.    15 days for training,
                        3 days for validation
    """
    first_initialized = False
    scaler_collection = None

    def __init__(self, data_path,
                 filename='sdwpf_baidukddcup2022_full.csv',
                 flag='train',
                 size=None,
                 turbine_id=0,
                 task='MS',
                 target='Patv',
                 scale=True,
                 start_col=3,       # the start column index of the data one aims to utilize
                 day_len=24 * 6,
                 train_days=15,     # 15 days
                 val_days=3,        # 3 days
                 total_days=30,     # 30 days
                 farm_capacity=134,
                 is_test=False,
                 tur_loc=[0., 0., 0.]
                 ):
        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        # initialization
        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]
        self.task = task
        self.target = target
        self.scale = scale
        self.start_col = start_col
        self.data_path = data_path
        self.filename = filename
        self.tid = turbine_id
        self.farm_capacity = farm_capacity
        self.is_test = is_test
        self.tur_loc = tur_loc
        # If needed, we employ the predefined total_size (e.g. one month)
        self.total_size = self.unit_size * total_days
        #
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        #
        if WindTurbineData.scaler_collection is None:
            WindTurbineData.scaler_collection = []
            for i in range(self.farm_capacity):
                WindTurbineData.scaler_collection.append(None)
        if self.is_test:
            if not WindTurbineData.first_initialized:
                self.__read_data__()
                WindTurbineData.first_initialized = True
        else:
            self.__read_data__()
        if not self.is_test:
            self.data_x, self.data_y = self.__get_data__(self.tid)
        

    def __read_data__(self):
        self.df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        self.df_raw.replace(to_replace=np.nan, value=0, inplace=True)
        
        self.df_raw.insert(3, column="loc_x", value=self.tur_loc[1])
        self.df_raw.insert(4, column="loc_y", value=self.tur_loc[2])
    def __get_turbine__(self, turbine_id):
        border1s = [turbine_id * self.total_size,
                    turbine_id * self.total_size + self.train_size - self.input_len
                    ]
        border2s = [turbine_id * self.total_size + self.train_size,
                    turbine_id * self.total_size + self.train_size + self.val_size
                    ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.task == 'MS':
            cols = self.df_raw.columns[self.start_col:]
            df_data = self.df_raw[cols]
            
        elif self.task == 'S':
            df_data = self.df_raw[[turbine_id, self.target]]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        cols = df_data.columns[2:]
        train_data = df_data[border1s[0]:border2s[0]]

        if WindTurbineData.scaler_collection[turbine_id] is None:
            preprocess(train_data)
            scaler = Scaler()
            scaler.fit(train_data[cols].values)
            WindTurbineData.scaler_collection[turbine_id] = scaler
        self.scaler = WindTurbineData.scaler_collection[turbine_id]
        res_data = df_data[border1:border2]
        preprocess(res_data)
        if self.scale:
            res_data[cols] = self.scaler.transform(res_data[cols].values)
        else:
            res_data = res_data.values
        return res_data.values

    def __get_data__(self, turbine_id):
        data_x = self.__get_turbine__(turbine_id)
        data_y = data_x
        return data_x, data_y

    def get_scaler(self, turbine_id):
        if self.is_test and WindTurbineData.scaler_collection[turbine_id] is None:
            self.__get_turbine__(turbine_id)
        return WindTurbineData.scaler_collection[turbine_id]

    def __getitem__(self, index):
        #
        # Rolling window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        # In our case, the rolling window is adopted, the number of samples is calculated as follows
        if self.set_type < 2:
            return len(self.data_x) - self.input_len - self.output_len + 1
        # Otherwise,
        return int((len(self.data_x) - self.input_len) / self.output_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def preprocess(data):

    data.fillna(method="bfill", inplace=True)
    data.fillna(method="ffill", inplace=True)
    data.reset_index(drop=True, inplace=True)

    unknown_1 = data.index[(data['Patv'] <= 0 ) & (data['Wspd'] > 2.5)]

    unknown_2 = data.index[(data['Pab1'] > 89) | (data['Pab2'] > 89) | (data['Pab3'] > 89)]
    pab1_err = data.index[data['Pab1'] > 89]
    for i in pab1_err:
        data['Pab1'][i] -= 89

    pab2_err = data.index[data['Pab2'] > 89]
    for i in pab2_err:
        data['Pab2'][i] -= 89
    
    pab3_err = data.index[data['Pab3'] > 89]
    for i in pab3_err:
        data['Pab3'][i] -= 89

    abnormal_1 = data.index[(data['Ndir'] > 720) | (data['Ndir'] < -720)]

    for i in abnormal_1:
        if data['Ndir'][i] > 0:   data['Ndir'][i] -= 1440
        else : data['Ndir'][i] += 1440

    abnormal_2 = data.index[(data['Wdir'] > 180) | (data['Wdir'] < -180)]

    for i in abnormal_2:
        if data['Wdir'][i] > 0:   data['Wdir'][i] -= 360
        else : data['Wdir'][i] += 360
    
    error_value = unknown_1
    error_value.union(unknown_2)
    error_value.union(abnormal_1)
    error_value.union(abnormal_2)

    i = 0
    while(i < len(error_value)):
        start = error_value[i]
        front = start - 1
        
        while(i+1 < len(error_value) and error_value[i] + 1 == error_value[i+1]):
            i += 1
    
        end = error_value[i]
        back = end + 1
        if back >= len(data):
            back = front 
            front -= 1
        
        if front == -1:
            front = back + 1
    
        value_dif = data['Patv'][back] - data['Patv'][front]
        len_dif = back - front
    
        for j in range(start, end+1):
            data['Patv'][j] = data['Patv'][front] + value_dif * (j - front) / len_dif 
    
        i+=1


def get_turbine_loc(loc_file_name):
    tur_loc = pd.read_csv(loc_file_name)
    cols = tur_loc.columns[1:]
    scaler = Scaler()
    scaler.fit(tur_loc[cols].values)
    tur_loc[cols] = scaler.transform(tur_loc[cols].values)

    tur_loc.to_csv("./normal_loc.csv",index=False)
    return tur_loc.values
            
