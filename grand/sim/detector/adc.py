"""
Master module for the ADC in GRAND
"""
import numpy as np
import logging
import scipy.fft as sf

logger = logging.getLogger(__name__)

class ADC:
    '''
    Class that represents the analog-to-digital converter (ADC) of GRAND.
    The ADC digitizes an analog voltage that has been processed through the entire RF chain.
    For GRAND, the ADC has:
    - a sampling rate of 500 MHz
    - 14 bits centered around 0 V <-> 0 ADC counts, with 13 positive and 13 negative bits
    - a saturation at an input voltage of +/- 0.9 V
    '''

    def __init__(self):
        self.sampling_rate = 500  # [MHz]
        self.max_bit_value = 8192 # 14 bit ADC;  2 x 2^13 bits for negative and positive ADC values
        self.max_voltage   = 9e5  # [µV]; saturation voltage of ADC (absolute value)


    def downsample(self,
                  voltage_trace,input_sampling_rate_mhz):
        '''
        downsamples the voltage trace to the target sampling rate
        
        Arguments
        ---------
        `voltage_trace`
        type        : np.ndarray[double]
        units       : uV 
        description : Array of voltage traces, with shape (N_du,3,N_samples)
                                    
        Returns
        -------
        `downsampled_voltage_trace`
        type        : np.ndarray[double]
        units       : uV
        description : Array of downsamplef voltage traces, with shape (N_du,3,N_samples)
        '''
        if self.sampling_rate != input_sampling_rate_mhz : 
          #compute the fft
          voltage_trace_f=sf.rfft(voltage_trace)
          #compute new number of points
          ratio=(self.sampling_rate/input_sampling_rate_mhz)        
          m=int(np.shape(voltage_trace)[2]*ratio)
          logger.info(f"resampling the voltage from {input_sampling_rate_mhz} to an ADC of {self.sampling_rate} MHz")        
          downsampled_voltage_trace=sf.irfft(voltage_trace,m)*ratio
        else:
          downsampled_voltage_trace=voltage_trace
        
        return downsampled_voltage_trace
        
    
    def _digitize(self,
                  voltage_trace):
        '''
        Performs the digitization of voltage traces at the ADC input:
        - converts voltage to ADC counts
        - quantizes the values
        
        Arguments
        ---------
        `voltage_trace`
        type        : np.ndarray[float]
        units       : µV
        description : Array of voltage traces at the ADC level, with shape (N_du,3,N_samples)
                                    
        Returns
        -------
        `adc_trace`
        type        : np.ndarray[int]
        units       : ADC counts (least significant bits)
        description : The digitized array of ADC traces, with shape (N_du,3,N_samples)
        '''
        
        # Convert voltage to ADC
        adc_trace = voltage_trace * self.max_bit_value / self.max_voltage

        # Quantize the trace
        adc_trace = np.trunc(adc_trace).astype(int)

        return adc_trace

    def _saturate(self,
                  adc_trace):
        '''
        Simulates the saturation of the ADC
        
        Arguments
        ---------
        `adc_trace`
        type        : np.ndarray[int]
        units       : ADC counts (least significant bits)
        description : Array of ADC traces, with shape (N_du,3,N_samples)
                                    
        Returns
        -------
        `saturated_adc_trace`
        type        : np.ndarray[int]
        units       : ADC counts (least significant bits)
        description : Array of saturated ADC traces, with shape (N_du,3,N_samples)
        '''
        
        saturated_adc_trace = np.where(np.abs(adc_trace)<self.max_bit_value,
                                       adc_trace,
                                       np.sign(adc_trace)*self.max_bit_value)

        return saturated_adc_trace
    
    def process(self,
                voltage_trace,
                noise_trace=None):
        '''
        Processes an analog voltage trace to a digital ADC trace,
        with an option to add measured noise
        
        Arguments
        ---------
        `voltage_trace`
        type        : np.ndarray[float]
        units       : µV
        description : Array of voltage traces at the ADC level, with shape (N_du,3,N_samples)

        `noise_trace` (optional)
        type        : np.ndarray[int]
        units       : ADC counts (least significant bits)
        description : Array of measured noise traces, with shape (N_du,3,N_samples)
        
        Returns
        -------
        `adc_trace`
        type        : np.ndarray[int]
        units       : ADC counts (least significant bits)
        description : Array of ADC traces with shape (N_du,3,N_samples)
        '''

        assert isinstance(voltage_trace,np.ndarray)       
          

        adc_trace = self._digitize(voltage_trace)

        # Add measured noise to the trace if requested
        if noise_trace is not None:
            assert isinstance(noise_trace,np.ndarray)
            assert noise_trace.shape == adc_trace.shape
            assert noise_trace.dtype == adc_trace.dtype
            adc_trace += noise_trace
            logger.info('Noise added to ADC trace')

        # Make sure the saturation occurs AFTER adding noise
        adc_trace = self._saturate(adc_trace)

        return adc_trace
