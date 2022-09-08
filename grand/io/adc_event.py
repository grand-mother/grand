'''
Created on Sep 5, 2022

@author: root
'''

from  grand.io.efield_antenna import MasterEfieldByAntenna

class MasterAdcEvent(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.efield = MasterEfieldByAntenna()
        
    def set_efield(self, o_efied):
        self.efield = o_efied
        pass
    
    def set_adc_event(self):
        pass
    
    def read(self, data_f):
        pass
    
    def write(self, data_f):
        pass
    
        