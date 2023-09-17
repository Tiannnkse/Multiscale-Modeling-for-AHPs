# -*- coding: utf-8 -*-
import numpy as np
import math
import re
from lmfit import Parameters, report_fit, Minimizer


class GET_ABC:
    """

    isothermal adsorption curve fitting

    """

    def __init__(self, T, xdata, ydata, xmax, ymax, list_abc_char, R=None):
        self.T = T
        self.R = R if R else 8.314
        self.xdata = xdata
        self.ydata = ydata
        self.xmax = xmax
        self.ymax = ymax
        self.list_abc_char = list_abc_char

    def fit_fc(self, params, xdata, ydata):

        A1 = params[self.list_abc_char[0]]
        A2 = params[self.list_abc_char[1]]
        A3 = params[self.list_abc_char[2]]

        B1 = params[self.list_abc_char[3]]
        B2 = params[self.list_abc_char[4]]
        B3 = params[self.list_abc_char[5]]

        C1 = params[self.list_abc_char[6]]
        C2 = params[self.list_abc_char[7]]
        C3 = params[self.list_abc_char[8]]

        def x_part(A, B, C, x):
            y_part_total = []
            for i in range(len(x)):
                try:
                    y_part = float(A) * math.pow(((x[i]) * math.exp(B)), C) / (1 + math.pow(((x[i]) * math.exp(B)), C))
                    y_part_total.append(y_part)
                except OverflowError:
                    y_part_total.append(-10)
            return y_part_total

        y1 = x_part(A1, B1, C1, xdata)
        y2 = x_part(A2, B2, C2, xdata)
        y3 = x_part(A3, B3, C3, xdata)

        model = []
        for j in range(len(xdata)):
            model_data = y1[j] + y2[j] + y3[j]
            model.append(model_data)
        model = np.array(model)
        return model - ydata

    def get_result(self):
        params = Parameters()
        params.add_many((self.list_abc_char[0], 0.55, True, 0, 1, None, None),
                        (self.list_abc_char[1], 0.15, True, 0, 1, None, None),
                        (self.list_abc_char[2], 0.3, True, 0, 1, '1-A1-A2', None),

                        (self.list_abc_char[3], 1, True, 0, np.inf, None, None),
                        (self.list_abc_char[4], 10, True, 0, np.inf, None, None),
                        (self.list_abc_char[5], 5, True, 0, np.inf, None, None),

                        (self.list_abc_char[6], 5, True, 0, np.inf, None, None),
                        (self.list_abc_char[7], 20, True, 0, np.inf, None, None),
                        (self.list_abc_char[8], 400, True, 0, np.inf, None, None),

                        )
        minner = Minimizer(self.fit_fc, params, fcn_args=(self.xdata, self.ydata), fcn_kws={})
        result = minner.scalar_minimize(method='Nelder-Mead', options={'maxiter': 500000000})
        # report_fit(result)
        result = str(result.params)
        return result

    def cal_abc(self, type_char, data):
        data = ''.join(data)
        data = data.strip(',')
        data = float(data)
        if type_char == 'A':
            return data
        elif type_char == 'B':
            return data * self.R * self.T
        elif type_char == 'C':
            return self.R * self.T / data

    def get_abc_dict(self):
        abc_dict = {}
        res = self.get_result()
        for val in self.list_abc_char:
            abc_dict[val] = re.findall(r"%s', value=(.+?) +" % val, res)
        return abc_dict

    def output_dict(self):
        abc_dict = self.get_abc_dict()
        output_dict = {}
        for val in self.list_abc_char:
            output_dict[val] = self.cal_abc(val[0], abc_dict[val])
        return output_dict
