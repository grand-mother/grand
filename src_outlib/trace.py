"""
prototype to store trace from io.root with numpy array
"""

import grand.io.root_trees as ioree
import numpy as np
import os.path

"""
    debut gerb : 
    sE=ior.EfieldEventTree('Coarse1.root')
    sE.get_list_of_events()
    sE.get_event(0)
    sE.du_id nom des detecteur
    sE.du_count nombre detecteur
    sE.pos_x position detecteur
    sE.du_nanoseconds date eventroot
    sE.trace_x efield sur x
    """


class TraceTableDC1(object):
    """
    use one structured numpy array ~ astropy Table
    """

    def __init__(self):
        """
        Constructor
        """
        self.unit_trace = ""
        self.dtype_table = np.dtype(
            [
                ("id_det", "i4"),
                ("tps", "f8"),
                ("x", "f4", (999,)),
                ("y", "f4", (999,)),
                ("z", "f4", (999,)),
            ]
        )
        # table of trace of event
        self.t_trace = None
        self.nb_events = -1
        # number detector in event
        self.nb_du = -1
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
        self.nb_du = self.ttree_evt.du_count
        print(self.t_trace.dtype, self.t_trace.shape)
        print(type(self.ttree_evt.du_id))
        print(self.ttree_evt.du_id)
        print(self.ttree_evt.trace_x)
        print(len(self.ttree_evt.trace_x))
        self.t_trace = np.zeros((self.nb_du,), dtype=self.dtype_table)
        self.t_trace["id_det"] = self.ttree_evt.du_id
        self.t_trace["tps"] = self.ttree_evt.du_nanoseconds
        self.t_trace["x"] = self.ttree_evt.trace_x
        self.t_trace["y"] = self.ttree_evt.trace_y
        self.t_trace["z"] = self.ttree_evt.trace_z
        return True


class EfieldDC1(object):
    """
    use sevral numpy array to store trace and id du , time
    """

    def __init__(self):
        """
        Constructor
        """
        self.unit_trace = ""
        # table of trace of event
        self.a_du = None
        self.a_tps = None
        self.a3_trace = None
        self.nb_events = -1
        # number detector in event
        self.nb_du = -1
        self.idx_event = -1

    def read_efield_event(self, f_name):
        if not os.path.exists(f_name):
            raise
        sE = ioree.EfieldEventTree(f_name)
        self.ttree_evt = sE
        # hard coding event number
        self.l_events = sE.get_list_of_events()
        self.nb_events = len(self.l_events)
        if len(self.l_events) == 0:
            raise
        self.idx_event = -1
        return self.next_event()
    
    def id_event(self):
        return self.l_events[self.idx_event][0]

    def next_event(self):
        self.idx_event += 1
        if self.idx_event >= self.nb_events:
            return False
        idx_event = self.l_events[self.idx_event]
        self.ttree_evt.get_event(idx_event[0], idx_event[1])
        self.nb_du = self.ttree_evt.du_count
        self.trace_size = len(self.ttree_evt.trace_x[0])
        self.t_du = np.array(self.ttree_evt.du_id, dtype=np.int16)
        self.t_tps = np.array(self.ttree_evt.du_nanoseconds, dtype=np.float32)        
        self.a3_trace = np.empty((self.nb_du, 3, self.trace_size), dtype=np.float32)
        self.a3_trace[:, 0, :] = np.array(self.ttree_evt.trace_x, dtype=np.float32)
        self.a3_trace[:, 1, :] = np.array(self.ttree_evt.trace_y, dtype=np.float32)
        self.a3_trace[:, 2, :] = np.array(self.ttree_evt.trace_z, dtype=np.float32)
        self.a3_pos = np.empty((self.nb_du, 3), dtype=np.float32)
        self.a3_pos[:, 0] = np.array(self.ttree_evt.pos_x, dtype=np.float32)
        self.a3_pos[:, 1] = np.array(self.ttree_evt.pos_y, dtype=np.float32)
        self.a3_pos[:, 2] = np.array(self.ttree_evt.pos_z, dtype=np.float32)
        return True

    def find_max_traces(self):
        """
        find absolute maximal vale in trace for each detector
        :param self:
        """
        return np.max(np.abs(self.a3_trace), axis=(1, 2))
