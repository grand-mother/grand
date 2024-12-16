#!/pbs/home/x/xtian/.conda/envs/grandlib2304/bin/python3.9
import numpy as np
import sys
import grand.dataio.root_trees as rt

def extract_trigger_parameters(trace, trigger_config, baseline=0):
    # Extract the trigger infos from a trace

    # Parameters :
    # ------------
    # trace, numpy.ndarray: 
    # traces in ADC unit
    # trigger_config, dict:
    # the trigger parameters set in DAQ

    # Returns :
    # ---------
    # Index in the trace when the first T1 crossing happens
    # Indices in the trace of T2 crossing happens
    # Number of T2 crossings
    # Q, Peak/NC

    # Find the position of the first T1 crossing
    index_t1_crossing = np.where((trace) > trigger_config["th1"],
                                 np.arange(len(trace)), -1)
    dict_trigger_infos = dict()
    mask_T1_crossing = (index_t1_crossing != -1)
    if sum(mask_T1_crossing) == 0:
        # No T1 crossing 
        raise ValueError("No T1 crossing!")
    dict_trigger_infos['index_T1_crossing'] = None
    # Tquiet to decide the quiet time before the T1 crossing 
    for i in index_t1_crossing[mask_T1_crossing]:
       # Abs value not exceeds the T1 threshold
       if i - trigger_config["t_quiet"]//2 < 0:
          raise ValueError("Not enough data before T1 crossing!")
       if np.all((trace[np.max([0, i - trigger_config['t_quiet'] // 2]):i]) <= trigger_config["th1"]):
          dict_trigger_infos["index_T1_crossing"] = i
          # the first T1 crossing satisfying the quiet condition
          break
    if dict_trigger_infos['index_T1_crossing'] == None:
       raise ValueError("No T1 crossing with Tquiet satified!")
    # The trigger logic works for the timewindow given by T_period after T1 crossing.
    # Count number of T2 crossings, relevant pars: T2, NCmin, NCmax, T_sepmax
    # From ns to index, divided by two for 500MHz sampling rate
    period_after_T1_crossing = trace[dict_trigger_infos["index_T1_crossing"]:dict_trigger_infos["index_T1_crossing"]+trigger_config['t_period']//2]
    # All the points above +T2
    positive_T2_crossing = (np.array(period_after_T1_crossing) > trigger_config['th2']).astype(int)
    # Positive crossing, the point before which is below T2.
    mask_T2_crossing_positive = np.diff(positive_T2_crossing) == 1
    # if np.sum(mask_T2_crossing_positive) > 0:
    #     index_T2_crossing_positive = np.arange(len(period_after_T1_crossing) - 1)[mask_T2_crossing_positive]
    negative_T2_crossing = (np.array(period_after_T1_crossing) < - trigger_config['th2']).astype(int)
    mask_T2_crossing_negative = np.diff(negative_T2_crossing) == 1
    # if np.sum(mask_T2_crossing_negative) > 0:
    #     index_T2_crossing_negative = np.arange(len(period_after_T1_crossing) - 1)[mask_T2_crossing_negative]
    # n_T2_crossing_negative = np.len(index_T2_crossing_positive)
    # Register the first T1 crossing as a T2 crossing
    mask_first_T1_crossing = np.zeros(len(period_after_T1_crossing), dtype=bool)
    mask_first_T1_crossing[0] = True
    # mask_first_T1_crossing[1:] = (mask_T2_crossing_positive | mask_T2_crossing_negative)
    mask_first_T1_crossing[1:] = (mask_T2_crossing_positive)
    index_T2_crossing = np.arange(len(period_after_T1_crossing))[mask_first_T1_crossing]
    n_T2_crossing = 1 # Starting from the first T1 crossing.
    dict_trigger_infos["index_T2_crossing"] = [0]
    if len(index_T2_crossing) > 1:
      for i, j in zip(index_T2_crossing[:-1], index_T2_crossing[1:]):
          # The separation between successive T2 crossings
          time_separation = (j - i) * 2
          if time_separation < trigger_config["t_sepmax"]:
              n_T2_crossing += 1
              dict_trigger_infos["index_T2_crossing"].append(j)
          else:
              # Violate the maximum separation, fail to trigger
              raise ValueError(f"Violating Tsepmax, the separation is {time_separation} ns.")
    else:
      n_T2_crossing = 1
      j = 1
    # Change the reference of indices of T2 crossing
    dict_trigger_infos["index_T2_crossing"] = np.array(dict_trigger_infos["index_T2_crossing"]) + dict_trigger_infos["index_T1_crossing"]
    dict_trigger_infos["NC"] = n_T2_crossing
    # Calulate the peak value
    dict_trigger_infos["Q"] = (np.max(np.abs(period_after_T1_crossing[:j])) - baseline) / dict_trigger_infos["NC"]
    return dict_trigger_infos

dict_trigger_parameter = dict([
  ("t_quiet", 512),
  ("t_period", 512),
  ("t_sepmax", 10),
  ("nc_min", 2),
  ("nc_max", 8),
  ("q_min", 0),
  ("q_max", 255),
  ("th1", 100),
  ("th2", 50),
  # Configs of readout timewindow
  ("t_pretrig", 960),
  ("t_overlap", 64),
  ("t_posttrig", 1024)
  ])


if __name__ == "__main__":
  # Read the traces from experimental data
  fname = sys.argv[1]
  file = rt.DataFile(fname)
  n_entries = file.tadc.get_number_of_entries()
  # Pad zeros at the head of the trace to statify the Tquiet condition
  zero_head = np.zeros(dict_trigger_parameter["t_quiet"] // 2, dtype=int)

  trigger_index = []
  # Loop over all entries
  for k in range(n_entries):
    file.tadc.get_entry(k)
    # Loop over four channels
    #for v in range(4):
    for v in range(3):
      trace = file.tadc.trace_ch[0][v]
      # Zero padding at first
      # trace_padded = np.concatenate((zero_head, trace))
      try:
        # Check if trigger
        trigger_infos = extract_trigger_parameters(trace, dict_trigger_parameter)
        # Save the triggered traces
        if trigger_infos["NC"] >= dict_trigger_parameter["nc_min"] and trigger_infos["NC"] <= dict_trigger_parameter["nc_max"]:
          trigger_index.append(k)
          break
      except ValueError:
        # No T1 crossing, no trigger
        # print(k, ": No trigger.")
        pass

  print(f"{fname}: {len(trigger_index)} out of {n_entries} triggered.")
  if len(trigger_index) > 0:
    np.savetxt(f"./{fname.split('/')[-1]}.trigger.txt", trigger_index, delimiter=', ', fmt='%d', header=str(dict_trigger_parameter))
