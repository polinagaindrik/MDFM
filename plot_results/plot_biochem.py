#!/usr/bin/env python3

import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm

if __name__ == "__main__":
    username = ''
    password = ''
    #df_ph_tb_poz, df_gas = read_biochem('experiments/biochemistry/', username, password)
    df_ph_tb_poz = fm.data.read_pkl('biochem_oh_tbars_poz.pkl', 'experiments/biochemistry/')
    df_gas = fm.data.read_pkl('biochem_o2_co2.pkl', 'experiments/biochemistry/')

    exp_temps = fm.output.read_from_json(''+'exp_temps.json', dir='inputs_fusionmodel/')
    df_gas = df_gas[df_gas.columns.drop(list(df_gas.filter(regex='M2')))]
    df_ph_tb_poz = df_ph_tb_poz[df_ph_tb_poz.columns.drop(list(df_ph_tb_poz.filter(regex='M2')))]
    
    fm.plotting.plot_biochem_measurement(df_gas, 'O2', exp_temps)
    fm.plotting.plot_biochem_measurement(df_gas, 'CO2', exp_temps)

    fm.plotting.plot_biochem_measurement(df_ph_tb_poz, 'pH', exp_temps)
    fm.plotting.plot_biochem_measurement(df_ph_tb_poz, 'TBARS', exp_temps)
    fm.plotting.plot_biochem_measurement(df_ph_tb_poz, 'POZ', exp_temps)