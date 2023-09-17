# -*- coding: utf-8 -*-

from sympy import *
from dealproject import Fitting
import math


class CAL_FUN(Fitting):
    # Multi-scale model (0D model)
    def __init__(self, para_dict, abc_dict, xdata, ydata, xmax, ymax):
        super(CAL_FUN, self).__init__(abc_dict, xdata, ydata, xmax, ymax)
        self.para_dict = para_dict

    def save_data(self, name_char, data_list):
        # Save the data of the cycle
        preheating_data = open(f'{name_char}', 'w+')
        for i in range(len(data_list[0][:])):
            output = str(data_list[0][i]) + ' ' \
                     + str(data_list[1][i]) + ' ' \
                     + str(data_list[2][i]) + ' ' \
                     + str(data_list[3][i]) + ' ' \
                     + str(data_list[4][i]) + "\n"
            preheating_data.write(output)

    def solve_haet(self, data_list, t, tt):
        # preheating process: isosteric phases
        def eq(t, tt):
            data_list[0].append(t)
            data_list[1].append(self.para_dict['T_s'])
            data_list[2].append(self.para_dict['rho_v'])
            data_list[3].append(self.para_dict['P'])
            data_list[4].append(self.para_dict['T_m'])

            eq_1 = (self.para_dict['rho_s'] * (1 - self.para_dict['epsilon']) * (
                        self.para_dict['C_s'] + self.para_dict['X'] * self.para_dict['C_pa']) + self.para_dict[
                        'epsilon'] * self.para_dict['rho_v'] * self.para_dict['C_pv']) * (
                               T_ss - self.para_dict['T_s']) / (
                           tt - t) - 4 * self.para_dict['d_outtube'] * self.para_dict['h_ms'] * (
                               self.para_dict['T_m'] - self.para_dict['T_s']) / (
                               self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2)
            eq_2 = (rho_vv - self.para_dict['rho_v']) / (tt - t) - 1 / (
                        self.para_dict['R_ethanol'] * self.para_dict['T_s'] ** 2) * (
                           (self.para_dict['T_s'] * (PP - self.para_dict['P']) / (tt - t)) - self.para_dict['rho_v'] * (
                               T_ss - self.para_dict['T_s']) / (tt - t))
            eq_3 = (PP - self.para_dict['P']) / (tt - t) - self.para_dict['P'] * self.para_dict['L_v'] / (
                        self.para_dict['R_ethanol'] * self.para_dict['T_s'] ** 2) * (T_ss - self.para_dict['T_s']) / (
                               tt - t)
            eq_4 = (self.para_dict['rho_m'] * self.para_dict['C_m']) * (T_mm - self.para_dict['T_m']) / (tt - t) - 4 * \
                   self.para_dict['d_in'] * self.h_fm(4.36) * (self.para_dict['T_f'] - self.para_dict['T_m']) / (
                           self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2) - 4 * self.para_dict['d_out'] * \
                   self.para_dict['h_ms'] * (self.para_dict['T_s'] - self.para_dict['T_m']) / (
                               self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2)
            result = linsolve([eq_1, eq_2, eq_3, eq_4], [T_ss, rho_vv, PP, T_mm])

            self.para_dict['T_s'] = float(list(result)[0][0])
            self.para_dict['rho_v'] = float(list(result)[0][1])
            self.para_dict['P'] = float(list(result)[0][2])
            self.para_dict['T_m'] = float(list(result)[0][3])
            
        T_ss, rho_vv, PP, T_mm = symbols('T_ss, rho_vv, PP, T_mm')
        while t < 30 and (self.para_dict['P'] < self.para_dict['P_con']):
            eq(t, tt)
            t += 0.01
            tt += 0.01

    def solve_des(self, data_list, t, tt):
        # desorption process
        def eq(t, tt):
            data_list[0].append(t)
            data_list[1].append(self.para_dict['X'])
            data_list[2].append(self.para_dict['T_s'])
            data_list[3].append(self.para_dict['rho_v'])
            data_list[4].append(self.para_dict['T_m'])

            eq_1 = (XX - self.para_dict['X']) / (tt - t) - self.K_LDF(self.para_dict['T_s']) * (
                        self.cal_fun(self.para_dict['T_s'], 10400, self.para_dict['R'], 'uptake') - self.para_dict['X'])
            eq_2 = (self.para_dict['rho_s'] * (1 - self.para_dict['epsilon']) * (
                        self.para_dict['C_s'] + self.para_dict['X'] * self.para_dict['C_pa']) + self.para_dict[
                        'epsilon'] * self.para_dict['rho_v'] * self.para_dict['C_pv']) * (
                               T_ss - self.para_dict['T_s']) / (tt - t) - (
                           1 - self.para_dict['epsilon']) * self.para_dict['rho_s'] * self.para_dict['H_ads'] * (
                               XX - self.para_dict['X']) / (tt - t) - 4 * self.para_dict['d_outtube'] * self.para_dict[
                       'h_ms'] * (self.para_dict['T_m'] - self.para_dict['T_s']) / (
                           self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2)
            eq_3 = (rho_vv - self.para_dict['rho_v']) / (tt - t) + self.para_dict['P'] / (
                        self.para_dict['T_s'] ** 2 * self.para_dict['R_ethanol']) * (T_ss - self.para_dict['T_s']) / (
                               tt - t)
            eq_4 = (self.para_dict['rho_m'] * self.para_dict['C_m']) * (T_mm - self.para_dict['T_m']) / (tt - t) - 4 * \
                   self.para_dict['d_in'] * self.h_fm(4.36) * (self.para_dict['T_f'] - self.para_dict['T_m']) / (
                           self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2) - 4 * self.para_dict['d_out'] * \
                   self.para_dict['h_ms'] * (self.para_dict['T_s'] - self.para_dict['T_m']) / (
                               self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2)
            result = linsolve([eq_1, eq_2, eq_3, eq_4], [XX, T_ss, rho_vv, T_mm])

            self.para_dict['X'] = float(list(result)[0][0])
            self.para_dict['T_s'] = float(list(result)[0][1])
            self.para_dict['rho_v'] = float(list(result)[0][2])
            self.para_dict['T_m'] = float(list(result)[0][3])

        XX, T_ss, rho_vv, T_mm = symbols('XX, T_ss, rho_vv, T_mm')
        while t < 300 and (self.para_dict['X'] - self.cal_fun(363, 10400, self.para_dict['R'], 'uptake')) > 1e-6:
            eq(t, tt)
            t += 0.01
            tt += 0.01

    def solve_cool(self, data_list, t, tt):
        # precooling process: isosteric phases
        def eq(t, tt):
            data_list[0].append(t)
            data_list[1].append(self.para_dict['T_s'])
            data_list[2].append(self.para_dict['rho_v'])
            data_list[3].append(self.para_dict['P'])
            data_list[4].append(self.para_dict['T_m'])

            eq_1 = (self.para_dict['rho_s'] * (1 - self.para_dict['epsilon']) * (
                        self.para_dict['C_s'] + self.para_dict['X'] * self.para_dict['C_pa']) + self.para_dict[
                        'epsilon'] * self.para_dict['rho_v'] * self.para_dict['C_pv']) * (
                               T_ss - self.para_dict['T_s']) / (
                           tt - t) - 4 * self.para_dict['d_outtube'] * self.para_dict['h_ms'] * (
                               self.para_dict['T_m'] - self.para_dict['T_s']) / (
                               self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2)
            eq_2 = (rho_vv - self.para_dict['rho_v']) / (tt - t) - 1 / (
                        self.para_dict['R_ethanol'] * self.para_dict['T_s'] ** 2) * (
                           (self.para_dict['T_s'] * (PP - self.para_dict['P']) / (tt - t)) - self.para_dict['rho_v'] * (
                               T_ss - self.para_dict['T_s']) / (tt - t))
            eq_3 = (PP - self.para_dict['P']) / (tt - t) - self.para_dict['P'] * self.para_dict['L_v'] / (
                        self.para_dict['R_ethanol'] * self.para_dict['T_s'] ** 2) * (T_ss - self.para_dict['T_s']) / (
                               tt - t)
            eq_4 = (self.para_dict['rho_m'] * self.para_dict['C_m']) * (T_mm - self.para_dict['T_m']) / (tt - t) - 4 * \
                   self.para_dict['d_in'] * self.h_fm(4.36) * (self.para_dict['T_f'] - self.para_dict['T_m']) / (
                           self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2) - 4 * self.para_dict['d_out'] * \
                   self.para_dict['h_ms'] * (self.para_dict['T_s'] - self.para_dict['T_m']) / (
                               self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2)
            result = linsolve([eq_1, eq_2, eq_3, eq_4], [T_ss, rho_vv, PP, T_mm])

            self.para_dict['T_s'] = float(list(result)[0][0])
            self.para_dict['rho_v'] = float(list(result)[0][1])
            self.para_dict['P'] = float(list(result)[0][2])
            self.para_dict['T_m'] = float(list(result)[0][3])

        T_ss, rho_vv, PP, T_mm = symbols('T_ss, rho_vv, PP, T_mm')
        while t < 30 and self.para_dict['P'] > self.para_dict['P_ev']:
            eq(t, tt)
            t += 0.01
            tt += 0.01

    def solve_ads(self, data_list, t, tt):
        # adsorption process
        def eq(t, tt):
            data_list[0].append(t)
            data_list[1].append(self.para_dict['X'])
            data_list[2].append(self.para_dict['T_s'])
            data_list[3].append(self.para_dict['rho_v'])
            data_list[4].append(self.para_dict['T_m'])

            eq_1 = (XX - self.para_dict['X']) / (tt - t) - self.K_LDF(self.para_dict['T_s']) * (
                        self.cal_fun(self.para_dict['T_s'], 4900, self.para_dict['R'], 'uptake') - self.para_dict['X'])
            eq_2 = (self.para_dict['rho_s'] * (1 - self.para_dict['epsilon']) * (
                        self.para_dict['C_s'] + self.para_dict['X'] * self.para_dict['C_pa']) + self.para_dict[
                        'epsilon'] * self.para_dict['rho_v'] * self.para_dict['C_pv']) * (
                               T_ss - self.para_dict['T_s']) / (tt - t) - (
                           1 - self.para_dict['epsilon']) * self.para_dict['rho_s'] * self.para_dict['H_ads'] * (
                               XX - self.para_dict['X']) / (tt - t) - 4 * self.para_dict['d_outtube'] * self.para_dict[
                       'h_ms'] * (self.para_dict['T_m'] - self.para_dict['T_s']) / (
                           self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2)
            eq_3 = (rho_vv - self.para_dict['rho_v']) / (tt - t) + self.para_dict['P'] / (
                        self.para_dict['T_s'] ** 2 * self.para_dict['R_ethanol']) * (T_ss - self.para_dict['T_s']) / (
                               tt - t)
            eq_4 = (self.para_dict['rho_m'] * self.para_dict['C_m']) * (T_mm - self.para_dict['T_m']) / (tt - t) - 4 * \
                   self.para_dict['d_in'] * self.h_fm(4.36) * (self.para_dict['T_f'] - self.para_dict['T_m']) / (
                           self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2) - 4 * self.para_dict['d_out'] * \
                   self.para_dict['h_ms'] * (self.para_dict['T_s'] - self.para_dict['T_m']) / (
                               self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2)
            result = linsolve([eq_1, eq_2, eq_3, eq_4], [XX, T_ss, rho_vv, T_mm])

            self.para_dict['X'] = float(list(result)[0][0])
            self.para_dict['T_s'] = float(list(result)[0][1])
            self.para_dict['rho_v'] = float(list(result)[0][2])
            self.para_dict['T_m'] = float(list(result)[0][3])

        T_ss, rho_vv, XX, T_mm = symbols('T_ss, rho_vv, XX, T_mm')
        while t < 300 and (self.cal_fun(303, 4900, self.para_dict['R'], 'uptake') - self.para_dict['X'] > 1e-6):
            eq(t, tt)
            t += 0.01
            tt += 0.01

    def cal_energy_Q(self, data_list_ph, data_list_ad, data_list_ds, data_list_pc, type_char=None):
        # Calculate the useful heat for each process
        type_char = type_char if type_char else None
        Q = 0
        if not type_char:
            print('Please provide calculation instructions')
            return None
        elif type_char == 'desorption':
            for i in range(len(data_list_ds[0][:]) - 1):
                Q_buff = (self.para_dict['rho_s'] * math.pi * (
                            self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2) / 4 * (
                                      self.para_dict['C_s'] + data_list_ds[1][i] * self.para_dict['C_pa']) +
                          self.para_dict['rho_m'] * math.pi * (
                                  self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2) / 4 * self.para_dict[
                              'C_m']) * (data_list_ds[2][i + 1] - data_list_ds[2][i]) - self.para_dict[
                             'rho_s'] * math.pi * (
                                 self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2) / 4 * self.para_dict[
                             'H_ads'] * (data_list_ds[1][i + 1] - data_list_ds[1][i])
                Q += Q_buff
        elif type_char == 'pre_cooling':
            for i in range(len(data_list_pc[0][:]) - 1):
                Q_buff = (self.para_dict['rho_s'] * math.pi * (
                            self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2) / 4 * (
                                      self.para_dict['C_s'] + data_list_ds[1][-1] * self.para_dict['C_pa']) +
                          self.para_dict['rho_m'] * math.pi * (
                                  self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2) / 4 * self.para_dict[
                              'C_m']) * (data_list_pc[1][i + 1] - data_list_pc[1][i])
                Q += Q_buff
        elif type_char == 'adsorption':
            for i in range(len(data_list_ad[0][:]) - 1):
                Q_buff = (self.para_dict['rho_s'] * math.pi * (
                            self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2) / 4 * (
                                      self.para_dict['C_s'] + data_list_ad[1][i] * self.para_dict['C_pa']) +
                          self.para_dict['rho_m'] * math.pi * (
                                  self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2) / 4 * self.para_dict[
                              'C_m']) * (data_list_ad[2][i + 1] - data_list_ad[2][i]) - self.para_dict[
                             'rho_s'] * math.pi * (
                                 self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2) / 4 * self.para_dict[
                             'H_ads'] * (data_list_ad[1][i + 1] - data_list_ad[1][i])
                Q += Q_buff
        elif type_char == 'pre_heating':
            for i in range(len(data_list_ph[0][:]) - 1):
                Q_buff = (self.para_dict['rho_s'] * math.pi * (
                            self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2) / 4 * (
                                      self.para_dict['C_s'] + data_list_ad[1][-1] * self.para_dict['C_pa']) +
                          self.para_dict['rho_m'] * math.pi * (
                                  self.para_dict['d_out'] ** 2 - self.para_dict['d_in'] ** 2) / 4 * self.para_dict[
                              'C_m']) * (data_list_ph[1][i + 1] - data_list_ph[1][i])
                Q += Q_buff
        return Q

    def cal_Qc(self, data_list_ds, data_list_ad):
        # energy Q_c
        Q_c = (self.para_dict['rho_s'] * math.pi * (
                    self.para_dict['d_outs'] ** 2 - self.para_dict['d_ins'] ** 2) / 4 * self.para_dict['H_v'] * (
                            data_list_ad[1][-1] - data_list_ds[1][-1]))
        return Q_c

    def h_fm(self, Nu_f):
        # Heat transfer coefficient between metal and adsorbent
        return Nu_f * self.para_dict['k_f'] / self.para_dict['d_in']

    def K_LDF(self, Ts):
        #  K_LDF constant is a function of temperature
        return self.para_dict['C'] * math.exp(-self.para_dict['Ea_0'] / (self.para_dict['R'] * Ts))

