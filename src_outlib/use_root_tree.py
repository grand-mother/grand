import os 
import time

import numpy as np

import grand.io.root_trees as groot

G_file_efield = "/home/dc1/Coarse1.root"

def do_read_write(f_name, f_out, read_before=True):
    #
    if read_before:
        print("I'm read EfieldEventTree")
        tt_efield = groot.EfieldEventTree(f_name)
        tt_efield.get_event(0)
        trace_x = tt_efield.trace_x
    # 
    tt_voltage = groot.VoltageEventTree(f_out)
    tt_voltage.run_number = 0  
    tt_voltage.event_number = 0
    tt_voltage.event_size = 1999
    for idx in range(10):
        trace = np.arange(tt_voltage.event_size, dtype=np.float64)
        tt_voltage.trace_x.append(trace.tolist())
        tt_voltage.trace_z.append(trace.tolist())
        tt_voltage.trace_z.append(trace.tolist())
    ret = tt_voltage.fill()
    #ret = tt_voltage.write(f_out, close_file=False)
    ret = tt_voltage.write()

def test_read_after_write(f_root, event_nb=0):
    o_vol = groot.VoltageEventTree(f_root)
    print(o_vol.get_list_of_events())
    try:
        o_vol.get_event(event_nb)
        print(o_vol.trace_x[0][:10])
        print("test_read_write is OK")       
    except:
        print("test_read_write is *** NOK ***")
        
        
def test_nok(f_out): 
    do_read_write(G_file_efield,f_out, read_before=True)
    test_read_after_write(f_out)
 
def remove_file_sleep(f_to_rm):
    try:
        os.remove(f_out)
        time.sleep(1)
    except:
        print(f"no file {f_to_rm}")
    
        
if __name__ == '__main__':
    f_out =  "voltage1.root"
    remove_file_sleep(f_out)
    test_nok(f_out)
    
    