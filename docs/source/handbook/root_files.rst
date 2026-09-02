.. This page is generated from resources/GRANDlib_Handbook.zip
   by docs/dev/build_handbook.py.  Do not edit it by hand.

ROOT Files
==========


In the GRAND framework, all simulated and experimental data are stored using the ``ROOT`` format, a widely adopted data analysis framework in high-energy and astroparticle physics.
ROOT files provide a hierarchical and compressed data structure that allows efficient storage, fast access, and flexible manipulation of complex event information.
They serve as the primary interface between detector simulations, data acquisition systems, and subsequent analysis pipelines.

Each ROOT file is organized into dedicated classes, where every class corresponds to a specific aspect of the experiment or simulation.
These classes encapsulate information about:

- **Run metadata and detector configuration**, including run numbers, acquisition modes, detector unit (DU) positions, orientations, hardware properties, and environmental conditions.

- **Signal-level data**, such as digitized traces from the analog-to-digital converters (ADCs), raw and processed voltages, and reconstructed electric fields.

- **Trigger information**, including local trigger patterns, rates, flags, and timing alignment across detector units.

- **Shower and simulation parameters**, covering the injected particle type and energy, shower core and axis reconstruction, atmosphere and magnetic field models, as well as simulation software versions and thinning factors.

- **Auxiliary information**, such as GPS synchronization, detector calibration constants, noise maps, and environmental monitoring data.

This structure ensures that ROOT files contain not only the physics events of interest but also the contextual metadata required to interpret, reproduce, and validate the data.
For instance, the same file may include detector-level raw traces (``TADC``), reconstructed voltage signals (``TVoltage``), the corresponding electric field at the antenna level (``TEfield``), as well as the shower properties that generated them (``TShower``).
Simulation runs are equally documented through specialized classes (``TRunEfieldSim``, ``TRunShowerSim``, ``TShowerSim``), ensuring reproducibility and comparability across different codes (e.g., ZHAireS, CoREAS).

   :width: 60.0%
   :width: 80.0%
   :width: 80.0%
   :width: 60.0%
   :width: 50.0%
   :width: 30.0%
   :width: 60.0%

