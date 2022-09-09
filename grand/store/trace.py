'''
simple event table 
'''
import grand.io.root.event.efield as ioree
import numpy as np


class TraceTableDC1(object):
    '''
    debut gerb : 
    sE=ior.EfieldEventTree('Coarse1.root')
    sE.get_list_of_events()
    sE.get_event(0)
    sE.du_id nom des detecteur
    sE.du_count nombre detecteur
    sE.pos_x position detecteur
    sE.du_nanoseconds date event
    sE.trace_x efield sur x
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.unit_trace = ""
        self.dtype_table = np.dtype([('id_det', 'i4'), ('tps', 'f8'), ('x', 'f4', (999,)), ('y', 'f4', (999,)), ('z', 'f4', (999,))])
        # table of trace of event
        self.t_trace = None
        self.nb_events = -1
        # number detector in event
        self.nb_det = -1 
        self.idx_event = -1

    def read_efield_event(self, f_name):
        sE = ioree.EfieldEventTree(f_name)
        self.ttree_evt = sE
        # hard coding event number
        self.l_events = sE.get_list_of_events()
        self.nb_events = len(self.l_events)
        if len(self.l_events) == 0:
            raise
        self.idx_event = -1
        return self.next_event()        
    
    def next_event(self):
        self.idx_event += 1
        if self.idx_event >= self.nb_events:
            return False
        idx_event = self.l_events[self.idx_event]
        self.ttree_evt.get_event(idx_event[0], idx_event[1])
        self.nb_det = self.ttree_evt.du_count
        self.t_trace = np.zeros((self.nb_det,), dtype=self.dtype_table)
        print(self.t_trace.dtype, self.t_trace.shape)
        print(type(self.ttree_evt.du_id))
        print(self.ttree_evt.du_id)
        print(self.ttree_evt.trace_x)
        print(len(self.ttree_evt.trace_x))
        self.t_trace['id_det'] = self.ttree_evt.du_id
        self.t_trace['tps'] = self.ttree_evt.du_nanoseconds
        self.t_trace['x'] = self.ttree_evt.trace_x
        self.t_trace['y'] = self.ttree_evt.trace_y
        self.t_trace['z'] = self.ttree_evt.trace_z
        return True
    
