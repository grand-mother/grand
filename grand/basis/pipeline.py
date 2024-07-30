# The above code defines a Python class `Pipeline` and a function `execute_pipeline` that are used in
# the pipeline execution.
"""
Functions used in the pipline execution.

RK TODO: implement techniques used in LP's execute_pipeline where output_trees
         and time-traces are saved on event-by-event basis.

@GRAND Collaboration, 2023
"""

import numpy as np
from logging import getLogger
import grand.dataio.root_trees as groot
from dataclasses import dataclass

logger = getLogger(__name__)

"""
Final usage:
from grand.io.pipeline import Pipeline

pipeline = Pipeline()
pipeline.Add("reader",f_input="f.root") 
pipeline.Add("coreas2root", ...)
pipeline.Add("efield2voltage", .....)
pipeline.Add("some_reco", ....)
pipeline.Add("something_else", ...)
pipeline.Add("writer", f_output="out.root")

"""
#idea: push trees of one events down the pipeline and append them to write at the end of the pipeline.
#      Repeat the same process for the next event.

# The `@dataclass` decorator is used to create a simple class to hold data. In this case, the
# `Pipeline` class is defined as a dataclass with two class variables `trees_dict` and `command_dict`.
# The `Add` method within the `Pipeline` class is used to add different components to the pipeline
# based on the `name` provided.
@dataclass
class Pipeline:

    trees_dict = {}
    command_dict = {}

    def Add(self, name, **kwargs):
        name = name.lower()
        if name=='reader':
            logger.info("Reading input files")
            if 'f_input' in kwargs.keys():
                self.f_input = kwargs['f_input']
            else:
                raise Exception("Provide an input filename with option f_input='<your_file.root>'. ")

            #print('filename:', self.f_input)
            #print('kwargs:', kwargs)

        if name=='efield2voltage':
            logger.info("Reading efield2voltage")
            # call class to compute voltage from efield.
            from grand.sim.efield2voltage import Efield2Voltage

            #print('kwargs:', kwargs)
            #events      = groot.EfieldEventTree(self.f_input) 
            #events_list = events.get_list_of_events()

            if 'seed' in kwargs.keys() and 'padding_factor' not in kwargs.keys():
                self.master = Efield2Voltage(self.f_input, seed=kwargs['seed'])
            if 'seed' not in kwargs.keys() and 'padding_factor' in kwargs.keys():
                self.master = Efield2Voltage(self.f_input, padding_factor=kwargs["padding_factor"])
            if 'seed' in kwargs.keys() and 'padding_factor' in kwargs.keys():
                self.master = Efield2Voltage(self.f_input, seed=kwargs['seed'], padding_factor=kwargs["padding_factor"])
            if 'seed' not in kwargs.keys() and 'padding_factor' not in kwargs.keys(): # use default seed=None, padding_factor=1.0
                self.master = Efield2Voltage(self.f_input)

            logger.info("master defined")

            if 'add_noise' in kwargs.keys():
                self.master.params['add_noise'] = kwargs['add_noise']
            if 'add_rf_chain' in kwargs.keys():
                self.master.params['add_rf_chain'] = kwargs['add_rf_chain']
            if 'lst' in kwargs.keys():
                self.master.params['lst'] = kwargs['lst']

            self.command_dict[name] = self.master

        if name=='writer':
            logger.info("writing on output file")

            for key, command in self.command_dict.items():
                if key=="efield2voltage":
                    command.f_output = kwargs['f_output']
                    command.compute_voltage()


'''
# ToDo: there should also be a kind of tree names list. Probably will be solved with usage of DataFile
def execute_pipeline(pipeline, filelist, output_dir=""):
    #Execute the pipeline

    # Changes the returns of functions to dictionaries - needed for the pipeline
    electronic_chain.ec_config.in_pipeline = True

    # Execute the prep_func before looping through files
    # Only one prep func allowed in pipeline
    if "prefileloop_call" in pipeline:
        # RK: prep_func returns {'rfft': True, 'irfft': True}. var_dict part has no effect at all. Why is this done?
        prep_func = importlib.__import__(pipeline["prefileloop_call"]["module"], fromlist=("prefileloop_call")).prep_func
        var_dict = prep_func(**pipeline["prefileloop_call"]["kwargs"]) # At this stage, var_dict={'rfft': True, 'irfft': True}

    # Loop through files
    for in_root_file in filelist:
        print("************** Analysing file", in_root_file)

        # Removing the trees from the previous file from memory
        # ToDo: This should not be up to a user, at least not in this ugly way
        grand.io.root_trees.grand_tree_list = []

        # Remove the previous traces, if exist
        if "traces_t" in var_dict:
            del var_dict["traces_t"]
            del var_dict["traces_f"]

        # Execute all the preeventloop_calls in the pipeline
        for (key,part) in pipeline.items():
            if part["type"]=="preeventloop_call":
                # Find the specified preeventloop function in the module
                print('preloop module, key', part["module"], key)
                func = getattr(importlib.__import__(part["module"]), key) # func = preevent_func inside PM_functions/preevent_func.py
                # Call the function. returns dictionary with to-be stored empty tree object, freqs_MHz, dt_ns, LNA, and filter values for freqs_MHz.
                var_dict.update(func(pipeline=pipeline, in_root_file=in_root_file, output_dir=output_dir, **var_dict))

        output_trees = var_dict["output_trees"]
        tshower, tefield = var_dict["tshower"], var_dict["tefield"]

        # Loop through events
        for i in range(tshower.get_entries()):
            tshower.get_entry(i)
            # ToDo: A bug in root_trees? Get entry should not be necessary after get entry on tshower (friends!)
            tefield.get_entry(i)
            # tvoltage.copy_contents(tefield)

            # Loop through the elements of the pipeline
            for (key,part) in pipeline.items():
                print("Applying ", key)
                # Skip the prefileloop_call and preeventloop_call
                # RK: maybe this should be --> if key=="prefileloop_call" or part["type"]=="preeventloop_call": continue
                #     There is no key called "preeventloop_call". There is a key called "preevent_func" though.
                if key=="prefileloop_call" or key=="preeventloop_call": continue

                # Take action depending on the type of the pipeline element
                # ToDo: when we upgrade to Python >=3.10, "match" conditional should be used
                if part["type"]=="call":
                    # ToDo: slightly more optimal to do the import before all the looping (but not much)
                    func = getattr(importlib.__import__(part["module"]), key)
                    # Merge the function arguments with the output dictionary
                    # ToDo: It should be dict union "|" for python >=3.9
                    if "kwargs" in part:
                        input_dict = {**part["kwargs"], **var_dict}
                    else:
                        input_dict = var_dict

                    res = func(**input_dict)
                    # Update the results dictionary
                    var_dict.update(res)
                    if "traces_t" in var_dict.keys():
                        print('final trace_t.shape:', var_dict["traces_t"].shape)

                    # RK
                    #if key=='generate_galacticnoise':
                    #    print('func:', func)
                    #    print('input_dict:', input_dict)
                    #    print('res:', res)
                    #    print('var_dict:', var_dict.keys())
                    #print('func:', func)
                    #print('key:', key)
                    #print('var_dict:', var_dict.keys())

                elif part["type"]=="add":
                    res = add_traces(addend=var_dict[key], **var_dict)
                    var_dict.update(res)

                elif part["type"]=="add_randomized":
                    #print('add_randomized:', key, var_dict[key])
                    res = add_traces_randomized(addend=var_dict[key], **var_dict)
                    var_dict.update(res)

                elif part["type"]=="multiply":
                    res = multiply_traces(multiplier=var_dict[key], **var_dict)
                    var_dict.update(res)

                # Store the results in a tree if requested
                elif part["type"]=="store":
                    if "copy_tefield" in part and part["copy_tefield"] == True:
                        store_traces(var_dict["traces_t"], part["output_tree"], tefield)
                    else:
                        store_traces(var_dict["traces_t"], part["output_tree"])

                # RK
                if key=='galactic_noise':
                    print("saved volt_fft_with_galnoise_LP.npy")
                    np.save("volt_fft_with_galnoise_LP.npy", var_dict["traces_f"])

        # Write all the trees that are to be written
        for tree in output_trees:
            print("Writing", tree.tree_name)
            tree.write()


def store_traces(traces_t, tree, copy_tree=None):
    """Stores provided traces in the provided tree"""
    # Copy contents of another tree if requested
    if copy_tree:
        tree.copy_contents(copy_tree)

    # Different traces fields for ADC tree
    if "ADC" in tree.type.upper():
        tree.trace_0 = traces_t[:, 0, :].astype(np.int16)
        tree.trace_1 = traces_t[:, 1, :].astype(np.int16)
        tree.trace_2 = traces_t[:, 2, :].astype(np.int16)
    else:
        tree.trace_x = traces_t[:, 0, :].astype(np.float32)
        tree.trace_y = traces_t[:, 1, :].astype(np.float32)
        tree.trace_z = traces_t[:, 2, :].astype(np.float32)

    tree.fill()
'''
