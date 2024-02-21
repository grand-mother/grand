#! /usr/bin/env python3

import numpy as np
np.set_printoptions(threshold = 3000)
from grand import ADC
import grand.dataio.root_trees as rt

'''
This will be an equivalent file to `convert_efield2voltage.py`. For now it's just a sandbox to test the ADC module.

Goal of this script:
- read voltage simulation file, containing a TVoltage tree with voltage traces processed through the RF chain
- convert analog voltage traces to digital ADC traces
- include an option to add measured noise to the ADC traces
- save the (noisy) ADC traces in a TADC tree
'''

f_input = '../sim2root/Common/sim_Xiaodushan_20221026_180000_RUN0_CD_DC2Alpha_0000/voltage_-1_L0_0000_with-rf_no-noise.root'
#f_input = '/sps/grand/tueros/DC2Alpha/GP300_Xi_Sib_Proton_1.53_82.1_101.7_8550/tvoltage_8550-8550_L0_0000_with-rf_no-noise.root'

df = rt.DataFile(f_input)
tvoltage = df.tvoltage
tvoltage.get_entry(0)

voltage_trace = np.array(tvoltage.trace)

adc = ADC()

adc_trace = adc.process(voltage_trace)

adc_trace_noise = adc.process(voltage_trace,noise_trace=adc_trace)
# adc_trace = adc.digitize(voltage_trace)
# print(adc_trace[10][0])
# adc_trace = adc.saturate(adc_trace)
# print(adc_trace[10][0])


print('done')
