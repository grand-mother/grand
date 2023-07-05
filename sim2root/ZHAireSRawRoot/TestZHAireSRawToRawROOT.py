#Testing to see if the contents of the example are correct.

import unittest
#from tests import TestCase
import os
import glob
import logging
logging.basicConfig(level=logging.DEBUG)
from ZHAireSRawToRawROOT import ZHAireSRawToRawROOT
import raw_root_trees as RawTrees
import AiresInfoFunctionsGRANDROOT as AiresInfo


#TODO: Test for multiple primaries
#TODO: Test for multiple runs
#TODO: Test for multiple events.

class ZHAireSRawFilesTest(unittest.TestCase):

    InputFolder="./GP10_192745211400_SD075V"
    OutputRawFile="GP10_192745211400_SD075V.root"
    RunId=0
    EventId=1
    EventName=""     #this is where the test will store the EventName  
    RawShowerTree=0  #this is where the test will store the RawShowerTree
    RawEfieldTree=0  #this is where the test will store the RawEfieldTree
    
     
    def setUp(self):
        self.test_input_event_exists()
        self.test_rawroot_file_production() #TODO:see how to avoid calling this every time 

    def test_input_event_exists(self):
        existance=os.path.isdir(self.InputFolder) 
        self.assertEqual(existance, True, "Test event folder does not exist")
        
    def test_rawroot_file_production(self):
        
        sryfile=glob.glob(self.InputFolder+"/*.sry")
        if(len(sryfile)==1):
          logging.debug("successfully retrieved a sry file")      
        else:
          logging.critical("there should be one and only one sry file in the input directory!. cannot continue!")
          return -1        

        if  not os.path.isfile(sryfile[0]) :
          logging.critical("Input Summary file not found, {}".format(sryfile))
          return -1
   
        EventName=AiresInfo.GetTaskNameFromSry(sryfile[0])              
        self.EventName=EventName
        #print("self", self.EventName)
        
        rawfile_exists=os.path.isfile(self.OutputRawFile)
        
        if(rawfile_exists):
          #logging.debug("removing existing root file")
          #os.remove(self.OutputRawFile)
          
          logging.debug("file exists, skipping creation") #TODO: this is to be removed once i learn how to run the setup only once.
          self.RawShowerTree=RawTrees.RawShowerTree(str(self.OutputRawFile)) #TODO: this is to be removed once i learn how to run the setup only once.
          self.RawEfieldTree=RawTrees.RawEfieldTree(str(self.OutputRawFile)) #TODO: this is to be removed once i learn how to run the setup only once.
          return 1
      
        ReturnedEventName=ZHAireSRawToRawROOT(self.OutputRawFile, self.RunId, self.EventId, self.InputFolder, TaskName="LookForIt", EventName="UseTaskName")

        self.RawShowerTree=RawTrees.RawShowerTree(str(self.OutputRawFile))
        self.RawEfieldTree=RawTrees.RawEfieldTree(str(self.OutputRawFile)) 
 
        self.assertEqual(EventName, ReturnedEventName, "Error producing event raw root file")
    
    def test_rawroot_file_exists(self):
        rawfile=glob.glob(self.OutputRawFile)
        self.assertEqual(self.OutputRawFile, rawfile[0], "OutputRawFile Not Found")        


    #this test will check that all the variables in the rawshowertree are what is on the event, using AiresInfo
    def test_rawshowertree(self):
        sryfile=glob.glob(self.InputFolder+"/*.sry")
        
        list_of_events=self.RawShowerTree.get_list_of_events()
        self.RawShowerTree.get_event(list_of_events[0][0])
        
        EventName=AiresInfo.GetTaskNameFromSry(sryfile[0]) 
        MyEventName=self.RawShowerTree.event_name
        self.assertEqual(EventName, MyEventName, "Event Name Test failed")
        
        Primary= AiresInfo.GetPrimaryFromSry(sryfile[0],"GRAND")   #TODO Test case of multiple primaries
        MyPrimary= self.RawShowerTree.primary_type[0]
        self.assertEqual(str(Primary), MyPrimary, "Primary Test failed")
        
        Zenith = float(AiresInfo.GetZenithAngleFromSry(sryfile[0],"Aires")) 
        MyZenith= self.RawShowerTree.zenith
        print(type(Zenith),type(MyZenith),Zenith-MyZenith)
        self.assertEqual(MyZenith, Zenith, "Zenith Test failed")
        
        
        
        
        
                 

if __name__ == '__main__':
    unittest.main()    
