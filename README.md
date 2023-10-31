# SCP_Simulation_of_MIL101Cr

main.py: mainly divided into 4 steps: preheating, desorption, precooling and adsorption. The cooling capacity(specific cooling power and coefficient of performance) of adsorption heat pump for MIL-101Cr/ethanol working pair is obtained.

101Cr.csv:experimental data for ethanol adsorption by MIL-101Cr at 318k, with the first column representing uptake and the second column representing pressure.

calfunction.py: The calculation process of the model is a function of four steps. Each step includes the solution of the adsorbent bed energy balance equation,the adsorbent bed energy balance equation, the adsorbent bed energy balance equation, and the metal tube energy balance equation.

getabc.py & dealproject.py: The function calculates the universal isotherm parameters according to the adsorption isotherm simulated by the universal isotherm model

adsorption.dat:


Large-scale Evaluation of Dynamic Performance in Adsorption Heat Pumps based on Metal-organic Frameworks.
The refrigeration performance of MIL-101Cr/ethanol working pair in AHPs was simulated to obtain the coefficient of performance (COP) and specific cooling power (SCP) values.
The initial parameter reference for the multi-scale model is from [1].
The experimental parameters reference is from [2].



References: [1]. Dias, J. M. S.; Costa, V. A. F., Which dimensional model for the analysis of a coated tube adsorber for
                adsorption heat pumps? Energy 2019, 174, 1110-1120.(https://doi.org/10.1016/j.energy.2019.03.028)
            [2]. Rezk, A.;  Al-Dadah, R.;  Mahmoud, S.; Elsayed, A., Investigation of Ethanol/metal organic frameworks for
                low temperature adsorption cooling applications. Applied Energy 2013, 112, 1025-1031.(http://dx.doi.org/10.1016/j.apenergy.2013.06.041)
