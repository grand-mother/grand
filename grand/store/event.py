'''
simple event table 
'''
import grand.io.root.event.efield as ioree

class EventTable(object):
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


    def __init__(self, params=""):
        '''
        Constructor
        '''
        self.unit = ""
        

    def read(self, data_f):
        self.data_f = data_f
        sE = ioree.EfieldEventTree(data_f)
        # hard coding event number
        sE.get_event(0)
        self.sE = sE
        
    
    def efield(self, du_id):
        pass
        
    def plot_efield(self, du_id):
        pass
        
    
    def write(self, data_f):
        pass
    