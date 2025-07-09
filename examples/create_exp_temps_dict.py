import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

import pandas as pd
import numpy as np

def define_exp(df_6h, df_12h):
    exp_temps = {'V01': 2., 'V02': 10., 'V03': 14., 'V04': 14., 'V05': 10., 'V06': 14.,
                 'V07': 2., 'V08': 2.,  'V09': 10.,
                 'V10': 2., 'V10-CCD01': df_6h, 'V10-CCD02': df_12h,
                 'V11': 2., 'V11-CCD02': df_12h,
                 'V12': 2., 'V12-CCD01': df_6h,
                 'V13': 2., 'V13-STOR01': 2., 'V14': 2.,      'V14-STOR01': 2.,
                 'V15': 2., 'V15-SLH01':  2.,  'V15-SLH02': 2.,
                 'V16': 2., 'V16-STOR01': 2.,
                 'V17': 2., 'V17-CCD02': df_12h,
                 'V18': 2., 'V18-CCD01': df_6h,
                 'V19': 2., 'V19-CCD02': df_12h,
                 'V20': 2., 'V20-CCD01': df_6h,
                 'V21': 2., 'V21-SLH03a': 2., 'V21-SLH03b': 2.,
                 'V22': 2., 
                 'V23': 2., 'V23-SLH04': 2., 'V23-SLH04-CCD01': df_6h,
                 'V24': 2., 'V24-SLH04': 2., 'V24-SLH04-CCD01': df_6h,
                 'V25': 2., 'V25-SLH04': 2., 'V25-SLH04-CCD01': df_6h,
                 'V26': 2., 'V26-SLH04': 2., 'V26-SLH04-CCD01-RP': df_6h, 'V26-SLH04-RP':2.,
                 'V27': 2., 'V27-SLH04': 2., 'V27-SLH04-CCD01-RP': df_6h, 'V27-SLH04-RP': 2.,
                 'V28': 2., 'V28-SLH04': 2., 'V28-SLH04-CCD01-RP': df_6h, 'V28-SLH04-RP': 2.,}
    return exp_temps

if __name__ == "__main__":
    temp_series_6h = fm.output.read_from_json('inputs_fusionmodel/input_6St_interruption.json')['temp']
    temp_series_12h = fm.output.read_from_json('inputs_fusionmodel/input_12St_interruption.json')['temp']
    d_6h = {'time': np.array(temp_series_6h)[:, 1], 'temperature': np.array(temp_series_6h)[:,0]}
    df_6h = pd.DataFrame(data=d_6h)
    d_12h = {'time': np.array(temp_series_12h)[:, 1], 'temperature': np.array(temp_series_12h)[:,0]}
    df_12h = pd.DataFrame(data=d_12h)
    exp_temps = define_exp(temp_series_6h, temp_series_12h)
    fm.output.json_dump(exp_temps, 'exp_temps.json', dir='inputs_fusionmodel/')