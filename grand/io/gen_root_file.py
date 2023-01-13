import os.path
from logging import getLogger

import numpy as np

import grand.io.root_trees as groot
from grand.basis.traces_event import Handling3dTracesOfEvent

logger = getLogger(__name__)


class FileVoltageEvent(FileEvent):
    """
    Goals of the class:
      * add access to event by index not by identifier (event, run)
      * initialize instance to the first event
      * conversion io.root array to numpy if necessary

    """

    def __init__(self, f_name):
        """
        Constructoreg
        """
        if not os.path.exists(f_name):
            raise
        logger.info(f"Events  in file {f_name}")
        event = groot.VoltageEventTree(f_name)
        super().__init__(event)
        self.load_event_idx(0)
        self.f_name = f_name

    def get_obj_handlingtracesofevent(self):
        """
        Return a traces container IO independent
        """
        o_tevent = super().get_obj_handlingtracesofevent()
        o_tevent.set_unit_axis("$\mu$V", "dir")
        return o_tevent


myfile=FileVoltageEvent("../../SIMUS/_Filter_EfieldVSignal_LST18_XDS_Stshp_0.117_22.8_0.0_vertical_radius5325.58_100resamples.root")