"""
Example script to compute voltage from electric-field using GRAND pipeline.
"""

from grand.io.pipeline import Pipeline

import grand.manage_log as mlg
logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger(log_stdout=True)
logger.info(mlg.string_begin_script())


pipeline = Pipeline()

pipeline.Add("reader", 
            f_input="/home/data_challenge1_pm_lwp/data/Coarse2_xmax_add.root") # filename = str, list of str

pipeline.Add("efield2voltage", 
            add_noise=True, 
            add_rf_chain=True, 
            lst=18,
            seed=0,
            padding_factor=1.2)

pipeline.Add("writer", 
            f_output="Coarse2_xmax_add_voltage_event.root")


logger.info(mlg.string_end_script())


'''
Final usage:

pipeline.Add("reader",f_input="f.root") 
pipeline.Add("coreas2root", ...)
pipeline.Add("efield2voltage", .....)
pipeline.Add("some_reco", ....)
pipeline.Add("something_else", ...)
pipeline.Add("writer", f_output="out.root")
'''