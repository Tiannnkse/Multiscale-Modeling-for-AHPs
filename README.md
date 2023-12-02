# SCP_Simulation_of_MIL101Cr
Performance Evaluation of Metal-organic Frameworks in Adsorption Heat Pumps via Multiscale Modeling

The refrigeration performance of MIL-101Cr/ethanol working pair in AHPs was simulated to obtain the coefficient of performance (COP) and specific cooling power (SCP) values.

The initial parameter reference for the multiscale modeling is from [1].
The experimental parameters reference is from [2].

`main.py`: mainly divided into 4 steps: preheating, desorption, precooling and adsorption. The cooling capacity (SCP and COP) of adsorption heat pump for MIL-101Cr/ethanol working pair is obtained.

`101Cr.csv`:experimental data for ethanol adsorption by MIL-101Cr at 318.15 K, with the first column representing uptake and the second column representing pressure.

`calfunction.py`: The calculation process of the model is a function of four steps. Each step includes the solution of the adsorbent bed energy balance equation,the adsorbent bed energy balance equation, the adsorbent bed energy balance equation, and the metal tube energy balance equation.

`getabc.py` & `dealproject.py`: The function calculates the universal isotherm parameters according to the adsorption isotherm simulated by the universal isotherm model[3].

`adsorption.dat` & `desorption.dat`: Storage of adsorption/desorption data, respectively representing: runing time, evaporating temperature of adsorbent, vapor density, temperature of Metal.

`precooling.dat` & `preheating.dat`: Storage of cooling/heating data, respectively representing: runing time, evaporating temperature of adsorbent, vapor density, evaporating pressure, temperature of Metal.

***
References:

[1]. Dias, J. M. S.; Costa, V. A. F., Which dimensional model for the analysis of a coated tube adsorber for
                adsorption heat pumps? Energy 2019, 174, 1110-1120.([]()https://doi.org/10.1016/j.energy.2019.03.028)
                
[2]. Rezk, A.;  Al-Dadah, R.;  Mahmoud, S.; Elsayed, A., Investigation of Ethanol/metal organic frameworks for
                low temperature adsorption cooling applications. Applied Energy 2013, 112, 1025-1031.([]()http://dx.doi.org/10.1016/j.apenergy.2013.06.041)
                
[3] K.C. Ng, M. Burhan, M.W. Shahzad, A.B. Ismail, A Universal Isotherm Model to Capture Adsorption Uptake and Energy Distribution of Porous Heterogeneous Surface, Scientific Reports 7(1) (2017) 10634. ([]()https://doi.org/10.1038/s41598-017-11156-6).
