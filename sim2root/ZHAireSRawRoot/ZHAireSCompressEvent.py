#!/usr/bin/env python3
import sys
import os
import tarfile    #for ...you wont believe it...taring the files
import logging    #for...you guessed it...logging
import argparse   #for command line parsing
import glob       #for listing files in directories
import subprocess #for launching subprocesses



#logging.debug('This is a debug message')
#logging.info('This is an info message')
#logging.warning('This is a warning message')
#logging.error('This is an error message')
#logging.critical('This is a critical message')
logging.basicConfig(level=logging.DEBUG)                    
 
def ZHAireSCompressEvent(eventdir, action):
    '''
    A function to clean and compress a ZHAireSRawEvent
    
    Parameters:
        eventdir (str): the directory containing the ZHAireS event, as it finished the sim
        action (str): action to perform: 
            archive: compress + delete,
            compress: store all necessary output files on a tgz
            uncompress or expand: uncompress the tgz
            delete: erese all the simulation files. Works only if a tgz is present (to be sure you are not deleting with no backup)
            clean: erase uneeded files (ZHAireS runner auxiliary files)
            help: show what each action does
    
    Returns:
        nothing if completed
        0 if no action taken
        -1 if error
    '''
    #provide alias
    if(action=="expand"):
      action="uncompress"
      
    #Check if it is not compressed already  
    if(action=="compress" or action=="archive" ):       
      
      # Find all files with .tgz extension in the current directory
      tgzs = glob.glob(eventdir+"/*.tgz")
      # Check if or more tgz files exists
      if len(tgzs) >= 1:
        # Get the file name without extension
        logging.critical("Error: A tgz file is already present. Are you sure the event is not already compressed?. Cannot continue")
        return 0  
      
      
    if(action=="compress" or action=="archive" or action=="delete"):     
      # Find all files with .idf extension in the current directory
      idfs = glob.glob(eventdir+"/*.idf")
      # Check if only one file exists
      if len(idfs) == 1:
        # Get the file name without extension
        filename = os.path.basename(idfs[0])  # get filename with extension
        name, ext = os.path.splitext(filename)  # split filename and extension
        JobName = name
      else:
        logging.critical("Error: No idf file or more than one idf file found. Cannot continue")
        return -1
        
      #we compress and save the .idf stores all the tables, and the sry can be regenerated from it if needed 
      # we compress and save the .inp, .lgf, .py, .stdout (for future reference)
      filenames=[]
      filenames.append(eventdir+"/"+JobName+".idf") 
      filenames.append(eventdir+"/"+JobName+".inp")
      filenames.append(eventdir+"/"+JobName+".lgf")
      filenames.append(eventdir+"/"+JobName+".stdout")
      filenames.append(eventdir+"/"+JobName+".sry")
      filenames.append(eventdir+"/"+JobName+".status")
      filenames.append(eventdir+"/"+JobName+".py")
      filenames.append(eventdir+"/"+JobName+".EventParameters")

      #add the Aires.status and the Aires.dirs
      filenames.append(eventdir+"/Aires.status")      
      filenames.append(eventdir+"/Aires.dirs")      

      #add the antpos.dat
      filenames.append(eventdir+"/antpos.dat")      

      #add the trace files
      for filename in glob.glob(eventdir+"/"+"*.trace"):
        filenames.append(filename)
      
      
      if(action=="compress" or action=="archive"):
        #here gos the compess things
        logging.info("compressing event %s" % name)
        tarfilename=eventdir+ "/" + JobName +".tgz"
        #if the file exist, then we dont continue
        if os.path.isfile(tarfilename) :
          logging.critical("Error: tgz file already exists, will not modify to play it safe: %s" % tarfilename)
          return -1    
      
        tar = tarfile.open(tarfilename,"w:gz")
      
      
        for filename in filenames:
          if os.path.isfile(filename) : 
            name = os.path.basename(filename)      
            tar.add(filename,arcname=name)
          else:
            logging.info("Could not include %s in the tar file. this might be normal if this is an Aires shower instead of a ZHAireS one. Check it up" % filename)

        tar.close()
      
      if(action=="archive" or action=="delete"):
         #before archiving or deleting, lets check that the tgz exists
         tarfilename=eventdir+ "/" + JobName +".tgz"
         #if the file exist, then we continue
         if os.path.isfile(tarfilename) :
           logging.info("%s files from event %s" % (action,JobName))
           for filename in filenames:
             try:    
               os.remove(filename)
             except:
               logging.info("notice: file %s do not exists, or could not be deleted. This might be ok if this is an Aires shower instead of a ZHAireS one. Check it up." % filename)                 
         else:
           logging.critical("Error: tgz file do not exists, will not delete things to play it safe: %s" % tarfilename)
           return -1   

    elif(action=="uncompress"):
      
      # Find all files with .tgz extension in the current directory
      tgzs = glob.glob(eventdir+"/*.tgz")
      # Check if only one file exists
      if len(tgzs) == 1:
        # Get the file name without extension
        filename = os.path.basename(tgzs[0])  # get filename with extension
        name, ext = os.path.splitext(filename)  # split filename and extension
        JobName = name
        logging.info("uncompressing event %s" % JobName)
      else:
        logging.critical("Error: No tgz file or more than one tgz file found. Cannot continue")
        return -1
        
      #here gos the compess things
      tarfilename=eventdir+ "/" + JobName +".tgz"
      
      #if the file exist, then we extract
      if os.path.isfile(tarfilename) :
          tar = tarfile.open(tarfilename,"r:gz")  
          tar.extractall(path=eventdir)
          # Restore the original modification times
          for member in tar.getmembers():
            os.utime(os.path.join(eventdir, member.name), (member.mtime, member.mtime))      
          tar.close()
          
          
      else:
        logging.critical("Error: tgz file not found or something" % tarfilename)
        return -1    

    elif(action=="clean"):
      logging.info("cleaning event in %s" % eventdir)
      #once the event status is "RunOK", we not longer need the JobId
      try:
        os.remove(eventdir+"/JobId")
      except:
        logging.debug("Error while trying to delete %s" % eventdir+"/JobId")    
      #nor the output from slurm (slurm*.out)
     
      for name in glob.glob(eventdir+"/slurm*.out"):
        try: 
          os.remove(name)
        except:
          logging.debug("Error while trying to delete",name)    
      
      # we delete the timefresnel-root.dat files.
      for name in glob.glob(eventdir+"/*timefresnel-root.dat"):
        try:
          os.remove(name)
        except:
          logging.debug("Error while trying to delete",name)   

     
      
    else:
      if(action!='help'):
        logging.critical('unrecognized action %s " % action')
      logging.critical('valid actions are:')
      logging.critical('archive: compress to a tgz and delete files') 
      logging.critical('compress: compress to a tgz and keep files')
      logging.critical('uncompress or expand: uncompress the tgz')     
      logging.critical('delete: remove simulation files, leaving the tgz file, that must exist')
      logging.critical('clean: remove unnecessary files')
  
 
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A script to clean and compress a ZHAireSRawEvent')
    parser.add_argument('EventDir', #Directory where the event is
                        metavar="eventdir", #name of the parameter value in the help
                        help='the directory containing the ZHAireS event, as it finished the sim') # help message for this parameter
    parser.add_argument('Action', #name of the parameter
                        metavar="action", #name of the parameter value in the help
                        help='action to perform: archive, compress, uncompress/expand, delete, clean, help') # help message for this parameter
                        
                        
    results = parser.parse_args()
    eventdir=results.EventDir
    action=results.Action
    
    ZHAireSCompressEvent(eventdir, action)
    

    

