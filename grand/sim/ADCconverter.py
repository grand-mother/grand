#!/usr/bin/env python3
# author: Jelena & Pablo

import numpy as np

# Step 1: Convert voltage to ADC
def voltage_to_adc(trace,
                     adc_sampling_rate=2,  # ns
                     adc_to_voltage=0.9e6/2**13):
    '''
    Description
    -----------
    Performs the virtual digitization of voltage traces at the ADC level:
    - downsamples the simulated signal to the ADC sampling rate.

    Parameters
    ----------
    trace : np.ndarray[float]
        Input voltage traces at the ADC level with dimensions (3, N_simu_samples).
        Units: µV.

    adc_sampling_rate : float, optional
        Sampling rate of the ADC. Default is 2 ns (500 MHz).
        Units: ns.

    adc_to_voltage : float, optional
        Conversion factor from ADC counts to voltage. Default is 0.9e6/2**13.
        Units: µV.

    Returns
    -------
    trace : np.ndarray[float]
        The digitized array of voltage traces, with the ADC sampling rate and in ADC counts.
        Units: LSB.
    '''

    # round the trace (by flooring)
    trace = np.floor(trace / adc_to_voltage)

    # Obtain downsampling factor
    downsampling_factor = 1 / adc_sampling_rate  # Adjusted to directly use adc_sampling_rate
    # for the standard values, this would be a factor of 1/4, so we "keep" every 4th sample
    if not downsampling_factor.is_integer():
        raise ValueError("Downsampling factor must be an integer. Please check your settings.")

    # Select all rows from the array and every downsampling_factor-th column, effectively downsampling the array along the second axis:
    downsampled_trace = trace[:, ::int(downsampling_factor)]

    return downsampled_trace



# Step 2: Pad the trace with a constant value and adjust its length to 2048
def padding(trace, padding_value=800):
    '''
    Description
    -----------
    Pads the trace with a constant value before the trace and adjusts its length to 2048.
    Converts the voltage traces to ADC counts.

    Arguments
    ---------
    `trace`
    type        : np.ndarray[float]
    units       : LSB
    description : The digitized array of voltage traces, with the ADC sampling rate and in ADC counts.

    `padding_value`
    type        : int or float
    units       : LSB
    description : Constant value used for padding the trace before.

    Returns
    -------
    `trace`
    type        : np.ndarray[float]
    units       : LSB
    description : The padded and adjusted array of voltage traces in ADC counts.
    '''
    # Pad the trace before with a constant value (800 in this case)
    trace = np.concatenate([np.full(padding_value, 800), trace])

    # Calculate the remaining padding needed after the trace
    padding_after = max(0, 2048 - len(trace))

    # Pad with zeros after the trace
    if padding_after > 0:
        trace = np.concatenate([trace, np.zeros(padding_after)])

    # Ensure the length is 2048
    trace = trace[:2048]

    # Return trace in ADC counts, with the ADC sampling rate, and peak in the center
    return trace

# Example usage:
# Assuming you have a trace
your_trace = np.random.rand(1000)  # Replace this with your actual trace

# Step 1: Sample reduction & conversion to ADC
downsampled_trace = voltage_to_adc(your_trace)

# Step 2: add padding
final_trace = padding(downsampled_trace)
