# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:37:55 2023

@author: Wei Li/Tiangui Liang/Zhiliang Cai
"""
import csv
import math
import pandas as pd
import numpy as np
import itertools
import logging
import getabc
import dealproject
import calfunction


def set_data(file_path):
    # data from adsorption isotherms ( T_isotherm = 318.15 K )
    file = csv.reader(open(file_path, 'r', encoding='UTF-8-sig'))

    # uptake and pressure(P/Ps)
    xdata = []
    ydata = []

    for line in file:
        xdata.append(float(line[0]))
        ydata.append(float(line[1]))

    ymax = max(ydata)
    xmax = max(xdata)
    ydata = np.array(ydata) / ymax

    return xdata, ydata, xmax, ymax


def cal_cop(Q_ev, Q_23, Q_34):
    # calculate COP value for cooling application
    COP = Q_ev / (Q_23 + Q_34)
    return COP


def cal_scp(Q_c, para_dict, data_list_des, data_list_cool, data_list_ads, data_list_heat):
    # calculate SCP value
    SCP = Q_c / (para_dict['rho_s'] * math.pi * (para_dict['d_outs'] ** 2 - para_dict['d_ins'] ** 2) / 4 * (
            data_list_des[0][-1] + data_list_cool[0][-1] + data_list_ads[0][-1] + data_list_heat[0][-1]))
    return SCP


if __name__ == '__main__':

    path = '101Cr.csv'

    xdata, ydata, xmax, ymax = set_data(path)

    # parameter dictionary:  R :gas constant  Ts:T_isotherm(K)
    para_dict = {'R': 8.314, 'T_iso': 318.15}

    # --------------Step 1:adsorption isotherm fitting--------------
    list_char = ['A', 'B', 'C']
    list_num = ['1', '2', '3']

    list_abc_char = ['{}{}'.format(i, j) for i, j in itertools.product(list_char, list_num)]

    get_abc = getabc.GET_ABC(para_dict['T_iso'], xdata, ydata, xmax, ymax, list_abc_char, para_dict['R'])
    abc_dict = get_abc.output_dict()
    dl = dealproject.Fitting(abc_dict, xdata, ydata, xmax, ymax)

    # --------------Step 2:initialization parameter------------
    # Multi-scale Modeling data
    para_dict['epsilon'] = 0.4  # Porosity
    para_dict['C_pa'] = 2460  # specific heat of ethanol
    para_dict['C_pv'] = 2460  # specific heat of Vapor
    para_dict['rho_v'] = 0.0895  # vapor density of ethanol by ideal gas state equation
    para_dict['d_outtube'] = 0.022  # Diameter of out,tube = Diameter of in,s (Adsorbent)
    para_dict['h_ms'] = 100  # Heat transfer coefficient between metal and adsorbent
    para_dict['k_f'] = 0.6  # Thermal conductivity of Fluid
    para_dict['d_outs'] = 0.023  # Diameter of out,s (Adsorbent)
    para_dict['d_ins'] = 0.022  # Diameter of in,s (Adsorbent)
    para_dict['d_out'] = 0.022  # Diameter of out (tube)
    para_dict['d_in'] = 0.02  # Diameter of in (tube)
    para_dict['rho_m'] = 2700  # Density of Metal
    para_dict['C_m'] = 910  # Specific heat of Metal
    para_dict['R_ethanol'] = 180.5  # Particular gas constant of ethanol
    para_dict['L_v'] = 2.4e6  # Tube length of Vaporization
    para_dict['H_v'] =  42.39 / 46 * 1e6 # Loading average enthalpy of ethanol vapor

    # working condition
    para_dict['P'] = 4900  # evaporating pressure
    para_dict['P_con'] = 10400  # condensing pressure
    para_dict['T_s'] = 303  # evaporating temperature of Adsorbent
    para_dict['T_m'] = 303  # Temperature of Metal
    para_dict['T_f'] = 363  # Regeneration temperature

    # MOFs data
    name = 'MIL-101Cr'
    para_dict['H_ads'] = 48.7 / 46 * 1e6  # Loading average enthalpy of adsorption
    para_dict['rho_s'] = 4397.41  # MIL-101Cr density from zeo++ (Kg.m-3)
    para_dict['C_s'] = 1000  # hypothetical heat capacity of MIL-101Cr
    para_dict['Ea_0'] = 47859.508  # activation energy
    para_dict['C'] = 527463.39  # C is a constant that is a function of adsorbent granule size

    # ------------Step 3:Start simulation of adsorption heat pump cycle---------------
    cf = calfunction.CAL_FUN(para_dict, abc_dict, xdata, ydata, xmax, ymax)

    # ------------pre_heating---------------
    t = 0
    tt = 0.01
    #  Adsorbed concentration in Equilibrium ( in the adsorbent )
    para_dict['X'] = dl.cal_fun(para_dict['T_s'], para_dict['P'], para_dict['R'], 'uptake')
    data_list_heat = [[], [], [], [], []]

    cf.solve_haet(data_list_heat, t, tt)
    name_data_list3 = '%s_preheating.dat' % name
    cf.save_data(name_data_list3, data_list_heat)
    logging.info("preheating over")

    # -------------desorption----------------
    t = 0
    tt = 0.01
    para_dict['T_s'] = data_list_heat[1][-1]
    para_dict['rho_v'] = data_list_heat[2][-1]
    para_dict['P'] = data_list_heat[3][-1]
    para_dict['T_m'] = data_list_heat[4][-1]
    data_list_des = [[], [], [], [], []]

    cf.solve_des(data_list_des, t, tt)
    name_data_list = '%s_desorption.dat' % name
    cf.save_data(name_data_list, data_list_des)
    logging.info("desorption over")

    # --------------pre_cooling-----------------------
    t = 0
    tt = 0.01
    para_dict['T_f'] = 303
    para_dict['P_ev'] = 4900
    para_dict['T_s'] = data_list_des[2][-1]
    para_dict['rho_v'] = data_list_des[3][-1]
    para_dict['T_m'] = data_list_des[4][-1]
    data_list_cool = [[], [], [], [], []]

    cf.solve_cool(data_list_cool, t, tt)
    name_data_list = '%s_precooling.dat' % name
    cf.save_data(name_data_list, data_list_cool)
    logging.info("precooling over")

    # -------------------adsorption--------------------
    t = 0
    tt = 0.01
    para_dict['T_s'] = data_list_cool[1][-1]
    para_dict['rho_v'] = data_list_cool[2][-1]
    para_dict['P'] = data_list_cool[3][-1]
    para_dict['T_m'] = data_list_cool[4][-1]
    data_list_ads = [[], [], [], [], []]

    cf.solve_ads(data_list_ads, t, tt)
    name_data_list = '%s_adsorption.dat' % name
    cf.save_data(name_data_list, data_list_ads)
    logging.info("adsorption over")

    # -------------------End of simulation--------------------

    # -------------------Energy calculation at each process--------------------

    # energy of adsorption
    Q_12 = cf.cal_energy_Q(data_list_heat, data_list_ads,
                           data_list_des, data_list_cool,
                           type_char='adsorption')
    # energy of pre_heating
    Q_23 = cf.cal_energy_Q(data_list_heat, data_list_ads,
                           data_list_des, data_list_cool,
                           type_char='pre_heating')
    # energy of desorption
    Q_34 = cf.cal_energy_Q(data_list_heat, data_list_ads,
                           data_list_des, data_list_cool,
                           type_char='desorption')
    # energy of pre_cooling
    Q_41 = cf.cal_energy_Q(data_list_heat, data_list_ads,
                           data_list_des, data_list_cool,
                           type_char='pre_cooling')

    # The useful cooling capacity (Qc) heats
    Q_c = cf.cal_Qc(data_list_des, data_list_ads)

    # calculate COP/SCP value for cooling application
    COP = cal_cop(Q_c, Q_23, Q_34)
    SCP = cal_scp(Q_c, para_dict, data_list_des, data_list_cool, data_list_ads, data_list_heat)

    print(f'SCP value:{SCP}, COP value:{COP}')

    # Saving the results
    results = pd.DataFrame(data={'name': name, 'SCP': SCP, 'COP': COP}, index=[0])
    results.to_csv('results.csv')


