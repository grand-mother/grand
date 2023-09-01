import numpy as np






def find_peak_time(signal, threshold):
    """Given a recorded signal and a threshold, find the time index where the peak from the cosmic rays signal arrives."""
    
    # Subtract the mean of the signal to remove any DC component
    signal = signal - np.mean(signal)

    # Apply a high-pass filter to remove any low-frequency noise
    b, a = signal.butter(4, 0.1, 'highpass')
    signal = signal.filtfilt(b, a)

    # Calculate the envelope of the signal using the Hilbert transform
    analytic_signal = signal.hilbert(signal)
    envelope = np.abs(analytic_signal)

    # Find the time index where the envelope of the signal exceeds the threshold
    peak_time = np.argmax(envelope > threshold)

    return peak_time
    
    
    


def determine_direction(ant_pos, signal):
    # Find the time delay between each pair of detectors
    num_detectors = len(ant_pos)
    delta_t = np.zeros((num_detectors, num_detectors))
    for i in range(num_detectors):
        for j in range(i+1, num_detectors):
            delta_t[i,j] = find_peak_time(signal[i]) - find_peak_time(signal[j])
            delta_t[j,i] = -delta_t[i,j]
    
    # Find the distance between each pair of detectors
    speed_of_light = 299792458.0  # Exact value of the speed of light in meters per second
    delta_r = speed_of_light * delta_t
    
    # Find the unit vectors pointing from each detector to the location of the cosmic ray
    unit_vecs = np.zeros((num_detectors, 3))
    for i in range(num_detectors):
        unit_vecs[i] = ant_pos[i] - ant_pos[0]
        unit_vecs[i] /= np.linalg.norm(unit_vecs[i])
    
    # Solve for the direction of the incoming cosmic ray
    A = np.dot(delta_r, unit_vecs)
    B = np.dot(unit_vecs.T, unit_vecs)
    t = np.dot(np.linalg.inv(B), A)
    t /= np.linalg.norm(t)
    
    # Convert the direction to spherical coordinates
    zenith = np.arccos(t[2])
    azimuth = np.arctan2(t[1], t[0])
    
    return zenith, azimuth

