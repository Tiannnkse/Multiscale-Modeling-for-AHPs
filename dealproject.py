# -*- coding: utf-8 -*-

import math


class Fitting():
    def __init__(self, abc_dict, xdata, ydata, xmax, ymax):
        self.abc_dict = abc_dict
        self.xdata = xdata
        self.ydata = ydata
        self.xmax = xmax
        self.ymax = ymax

    def W_part(self, a, b, c, T, P, R):

        def exp_P(T):
            if T <= 353:
                return 133.32 * math.pow(10, (8.20417 - (1642.89 / ((T - 273) + 230.3))))
            else:
                return 133.32 * math.pow(10, (7.68117 - (1332.04 / ((T - 273) + 199.2))))

        try:
            W_part = a * math.pow(((P / exp_P(T)) * math.exp(b / (R * T))), (R * T / c)) / (
                    1 + math.pow(((P / exp_P(T)) * math.exp(b / (R * T))), (R * T / c)))
        except ValueError:
            W_part = 0
        except OverflowError:
            W_part = 0
        return W_part

    def cal_fun(self, T, P, R, type_char=None, W=None):
        type_char = type_char if type_char else None
        W = W if W else None
        uptake = self.W_part(self.abc_dict['A1'], self.abc_dict['B1'], self.abc_dict['C1'], T, P, R) + \
                 self.W_part(self.abc_dict['A2'], self.abc_dict['B2'], self.abc_dict['C2'], T, P, R) + \
                 self.W_part(self.abc_dict['A3'], self.abc_dict['B3'], self.abc_dict['C3'], T, P, R)
        if type_char == 'uptake':
            return uptake * self.ymax
        elif type_char == 'error':
            return uptake - W
        else:
            print('Instruction not found')
            return 'Instruction not found'

