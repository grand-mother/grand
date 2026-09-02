The simulation chain
====================

.. contents::
   :local:

From an electric field produced by an external air-shower simulation to the
digitized voltage a detection unit records.  This is what GRANDlib is for; the
physics upstream of it is not.

The stages
----------

.. code-block:: text

    ZHAireS / CoREAS          external: shower and radio emission
        |
        v
    sim2root                  simulator output -> GRANDROOT trees
        |
        v
    E-field  --> V_oc         antenna effective length
             --> + noise      Galactic background
             --> RF chain     LNA, baluns, cable, VGA, filter
             --> ADC          digitisation
        |
        v
    TVoltage / TADC trees

Open-circuit voltage
--------------------

The response of the antenna to an incoming field is its **effective length**
:math:`\boldsymbol{\ell}`, a vector that depends on direction and frequency.
The open-circuit voltage at one arm is its projection onto the field:

.. math::

   V_{\mathrm{oc}}^{p} = \boldsymbol{\ell}^{\,p} \cdot \boldsymbol{E}
     = \ell^{p}_{x} E_{x} + \ell^{p}_{y} E_{y} + \ell^{p}_{z} E_{z},

with :math:`p` running over the three arms.  The computation is done in the
frequency domain over 30–250 MHz.  :mod:`grand.sim.detector.process_ant`
interpolates the tabulated response in frequency, azimuth and zenith.

Galactic noise
--------------

Sky brightness temperatures from LFMap, folded through the antenna response
and tabulated over frequency and local sidereal time.  Each detection unit
receives an independent realisation; spatial coherence across the array is not
modelled.

.. jupyter-execute::

    import numpy as np
    from grand.sim.noise.galaxy import galactic_noise

    freqs = np.arange(30.0, 251.0)
    v = galactic_noise(18.0, 1024, freqs, nb_ant=4, seed=0)
    print("shape (units, arms, frequencies):", v.shape)

.. warning::

   The normalisation of this is an open question — the simulated RMS does not
   reproduce the tabulated model, and the change proposed to fix it does not
   fully reconcile them either.  See
   :ref:`issue-galactic-noise-normalisation` before relying on absolute noise
   levels.

The RF chain
------------

A cascade of two-port networks: low-noise amplifier, balun, cable and
connector, variable-gain amplifier with filter, a second balun, and the ADC
input.  Each is characterised by measured scattering parameters, converted to
the transmission (ABCD) representation so that the cascade is a matrix
product.

.. jupyter-execute::

    import numpy as np
    from grand.sim.detector.rf_chain import RFChain

    chain = RFChain(vga_gain=20)
    chain.compute_for_freqs(np.arange(30.0, 251.0))
    tf = np.abs(chain.get_tf())
    print("transfer function shape (arms, frequencies):", tf.shape)
    print("peak |V_out/V_oc| per arm:", np.round(tf.max(axis=1), 1))

The gain is a configuration choice: S-parameters are shipped for VGA gains of
20 dB (the GRANDProto300 default), 5 dB, 0 dB and −5 dB.

Digitisation
------------

:class:`~grand.sim.detector.adc.ADC` applies the sampling, quantisation and
saturation of the ADC chip, producing the counts a ``TADC`` tree holds.

Running it
----------

.. code-block:: python

    from grand import Efield2Voltage

    signal = Efield2Voltage("input_efield.root", "output_voltage.root")
    signal.params["add_noise"]    = True
    signal.params["add_rf_chain"] = True
    signal.compute_voltage()

or:

.. code-block:: bash

    python scripts/convert_efield2voltage.py in.root -o out.root --lst 18
    python scripts/convert_voltage2adc.py out.root -o adc.root

Run time is about 13 s per shower across a full GRANDProto300 array on one
core, measured over 300 ZHAireS showers for the GRANDlib paper.

What is not here
----------------

No trigger.  The library records trigger information in its trees but does not
decide whether a unit would have triggered, which is the step between "a
signal is present in this trace" and "this event was recorded" — and therefore
the step between a simulated trace and a sensitivity.

No reconstruction of shower parameters from recorded voltages; ``grand/recon``
is a stub.

No anthropogenic or radio-frequency-interference background; only the Galactic
component is modelled, and measured noise traces are used where a realistic
background is needed.
