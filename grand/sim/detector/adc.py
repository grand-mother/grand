"""
Master module for the ADC in GRAND
"""
import numpy as np

from grand.basis import validate as _validate
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
        r"""Creates the ADC model.

        Sets the bit depth, sampling rate and saturation level of the chip
        used by a GRAND detection unit.
        """
        self.sampling_rate = 500  # [MHz]
        self.max_bit_value = 8192 # 14 bit ADC;  2 x 2^13 bits for negative and positive ADC values
        self.max_voltage   = 9e5  # [µV]; saturation voltage of ADC (absolute value)


    def downsample(self,
                  voltage_trace,input_sampling_rate_mhz):
        '''
        downsamples the voltage trace to the target sampling rate

        Parameters
        ----------
        voltage_trace : np.ndarray[double]
            Array of voltage traces, with shape (N_du,3,N_samples), in µV.
        input_sampling_rate_mhz : float
            Sampling rate of `voltage_trace`, in MHz.

        Returns
        -------
        downsampled_voltage_trace : np.ndarray[double]
            Array of downsampled voltage traces, with shape (N_du,3,N_samples), in µV.

        Examples
        --------
        Simulated voltages are sampled at 2 GHz; the ADC samples at 500 MHz:

        .. jupyter-execute::

            import numpy as np
            from grand import ADC

            voltage = np.zeros((1, 3, 8192))                     # one unit, 4.096 µs at 2 GHz
            print(ADC().downsample(voltage, 2000.0).shape)
        '''
        # A rate of 0, NaN or below 0 failed with a bare ZeroDivisionError or
        # a negative dimension (#289)
        input_sampling_rate_mhz = _validate.as_real(input_sampling_rate_mhz, "input_sampling_rate_mhz",
                                                    "ADC.downsample")
        _validate.positive(input_sampling_rate_mhz, "input_sampling_rate_mhz", "ADC.downsample", "MHz")
        _validate.plausible(input_sampling_rate_mhz, "input_sampling_rate_mhz", "ADC.downsample", "sampling_rate_mhz")   # (#266)
        if self.sampling_rate != input_sampling_rate_mhz : 
          #compute the fft
          voltage_trace_f=sf.rfft(voltage_trace)
          #compute new number of points
          ratio=(self.sampling_rate/input_sampling_rate_mhz)        
          # Rounded, not truncated: 999 samples at 1 GHz gave 499 at 500 MHz (#289)
          m=int(round(np.shape(voltage_trace)[2]*ratio))
          logger.info(f"resampling the voltage from {input_sampling_rate_mhz} to an ADC of {self.sampling_rate} MHz")        
          downsampled_voltage_trace=sf.irfft(voltage_trace_f,m)*ratio
          #plt.plot(np.arange(0,len(downsampled_voltage_trace[0][0]))/ratio,downsampled_voltage_trace[0][0])
          #plt.plot(np.arange(0,len(voltage_trace[0][0])),voltage_trace[0][0])
          #plt.show()
        else:
          downsampled_voltage_trace=voltage_trace
        
        return downsampled_voltage_trace
        
    
    def _digitize(self,
                  voltage_trace):
        '''
        Performs the digitization of voltage traces at the ADC input:
        - converts voltage to ADC counts
        - quantizes the values

        Parameters
        ----------
        voltage_trace : np.ndarray[float]
            Array of voltage traces at the ADC level, with shape (N_du,3,N_samples), in µV.

        Returns
        -------
        adc_trace : np.ndarray[int]
            The digitized array of ADC traces, with shape (N_du,3,N_samples), in ADC counts.

        '''
        
        # Convert voltage to ADC
        adc_trace = voltage_trace * self.max_bit_value / self.max_voltage

        # Bounded in floating point before the integer cast: a value beyond the
        # int64 range became its most negative value, whose absolute value is
        # negative too, so saturation never caught it (#239).  The bound is far
        # beyond saturation, which _saturate() applies after any added noise.
        bound = float(2 ** 40)
        adc_trace = np.clip(adc_trace, -bound, bound)

        # Quantize the trace
        adc_trace = np.trunc(adc_trace).astype(int)

        return adc_trace

    def _saturate(self,
                  adc_trace):
        '''
        Simulates the saturation of the ADC

        Parameters
        ----------
        adc_trace : np.ndarray[int]
            Array of ADC traces, with shape (N_du,3,N_samples), in ADC counts.

        Returns
        -------
        saturated_adc_trace : np.ndarray[int]
            Array of saturated ADC traces, with shape (N_du,3,N_samples), in ADC counts.

        '''
        
        saturated_adc_trace = np.where(np.abs(adc_trace)<self.max_bit_value,
                                       adc_trace,
                                       np.sign(adc_trace)*self.max_bit_value)

        # Saturation is reported, not applied silently (#239)
        clipped = np.abs(adc_trace) >= self.max_bit_value
        if clipped.any():
            per_du = clipped.reshape(clipped.shape[0], -1).sum(axis=1) if clipped.ndim > 1 else [clipped.sum()]
            logger.warning("ADC saturation: %d samples clipped at +/-%d counts, in %d of %d units "
                           "(per unit: %s)", int(clipped.sum()), self.max_bit_value,
                           int(np.count_nonzero(per_du)), len(per_du),
                           [int(n) for n in per_du][:20])

        return saturated_adc_trace
    
    def process(self,
                voltage_trace,
                noise_trace=None):
        '''
        Processes an analog voltage trace to a digital ADC trace,
        with an option to add measured noise

        Parameters
        ----------
        voltage_trace : np.ndarray[float]
            Array of voltage traces at the ADC level, with shape (N_du,3,N_samples), in µV.

        noise_trace : np.ndarray[int], optional
            Array of measured noise traces, with shape (N_du,3,N_samples), in
            ADC counts.

        Returns
        -------
        adc_trace : np.ndarray[int]
            Array of ADC traces with shape (N_du,3,N_samples), in ADC counts.

        Examples
        --------
        A signal below one count does not come out small, it comes out **absent**.
        A simulation whose voltages land under the step yields an all-zero trace,
        which reads as a quiet event rather than a scaling error.

        .. jupyter-execute::

            import numpy as np
            from grand.sim.detector.adc import ADC

            adc = ADC()
            lsb = adc.max_voltage / adc.max_bit_value
            print("one count is %.1f uV" % lsb)

            t = np.arange(512) * 0.5
            pulse = np.exp(-((t - 100.0) ** 2) / (2 * 5.0 ** 2))

            for amplitude in (lsb / 10, lsb * 100):
                trace = np.stack([np.stack([pulse * amplitude] * 3)])
                print("%9.1f uV -> %4d counts"
                      % (amplitude, int(np.abs(np.asarray(adc.process(trace))).max())))
        '''

        if not isinstance(voltage_trace, np.ndarray):
            raise TypeError(_validate.message(
                "ADC.process", "'voltage_trace' must be a NumPy array, got %s"
                % type(voltage_trace).__name__))
        # NaN or inf cannot be digitized: it was written as the most negative
        # integer, and the error came only later, from the tree (#239)
        if not np.all(np.isfinite(voltage_trace)):
            where = np.argwhere(~np.isfinite(voltage_trace))
            raise ValueError(_validate.message(
                "ADC.process", "'voltage_trace' has %d NaN or infinite samples, first at "
                "(unit, channel, sample) index %s" % (len(where), tuple(int(i) for i in where[0]))))

        adc_trace = self._digitize(voltage_trace)

        # Add measured noise to the trace if requested
        if noise_trace is not None:
            if not isinstance(noise_trace, np.ndarray):
                raise TypeError(_validate.message(
                    "ADC.process", "'noise_trace' must be a NumPy array, got %s"
                    % type(noise_trace).__name__))
            if noise_trace.shape != adc_trace.shape:
                raise ValueError(_validate.message(
                    "ADC.process", "'noise_trace' must have the shape of the digitized trace, "
                    "%s, got %s" % (adc_trace.shape, noise_trace.shape)))
            if noise_trace.dtype != adc_trace.dtype:
                raise TypeError(_validate.message(
                    "ADC.process", "'noise_trace' must be in ADC counts (%s), got %s"
                    % (adc_trace.dtype, noise_trace.dtype)))
            adc_trace += noise_trace
            logger.info('Noise added to ADC trace')

        # Make sure the saturation occurs AFTER adding noise
        adc_trace = self._saturate(adc_trace)

        return adc_trace
