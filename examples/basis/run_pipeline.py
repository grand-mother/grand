#!/usr/bin/env python3

"""
Example script to compute voltage from electric-field using GRAND pipeline.
"""

from grand.basis.pipeline import Pipeline

import grand.manage_log as mlg
logger = mlg.get_logger_for_script(__file__)
mlg.create_output_for_logger(log_stdout=True)
logger.info(mlg.string_begin_script())

logger.info("Example script to implement pipeline.")

pipeline = Pipeline()

pipeline.Add("reader", 
            f_input="../data/test_efield.root") # filename = str, list of str

pipeline.Add("efield2voltage", 
            add_noise=True, 
            add_rf_chain=True, 
            lst=18,
            seed=0,
            padding_factor=1.2)

pipeline.Add("writer", 
            f_output="test_voltage.root")


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