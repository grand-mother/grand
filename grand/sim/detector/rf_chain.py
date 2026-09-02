#!/usr/bin/env python
import os
import xml.etree.ElementTree as ET
import os.path
import numpy as np
from pathlib import Path

from grand import grand_add_path_data
from logging import getLogger
logger = getLogger(__name__)

"""
RF Chain Simulation with XML Configuration (modified by SN)

This script simulates the RF chain by processing a series of electronic components 
(e.g., balun, matching network, LNA, VGA, AD chip). Previously, component file paths 
were hardcoded. Now, paths are dynamically loaded from an external XML configuration 
(`config.xml`), improving flexibility and maintainability.

Key Updates:
- Replaced hardcoded file paths with XML-based configuration.
- Supports dynamic selection of `.s2p`, `.s1p`, and `.csv` files.
- Handles different axes (`X`, `Y`, `Z`) dynamically.
- Improved error handling and debugging.

Developed & tested in `dev_snonis` branch. Final validation in progress before merging into `dev`.
"""

"""
RF Chain (version 2) of both detector units at the GP13 site in Dunhuang and G@Auger in Malargue.

This code includes: 
 * Antenna Impedance
 * Input Impedance  (computed after computing total_ABCD_matrix)
 * Balun before matching_network
 * XYZ_matching network
 * new LNA
 * cable + connector
 * VGA + Filter
 * Balun after VGA&filter + 200 ohm load + AD Chip

:Authors: PengFei Zhang, Xu Xin and Xidian University group, GRANDLIB adaptation Colley JM, Ramesh, and Sandra.
:Modified by SN including rf chain v2
Reference document: 
  * "RF Chain simulation for GP300" by Xidian University, :Authors:  Pengfei Zhang and Pengxiong Ma, Chao Zhang, Rongjuan Wand, Xin Xu
  
Convention Port/direction:
 * 0 for X / North, South
 * 1 for Y / East/West
 * 2 for Z / Up
 
# Note: plotting has been moved to grand/scripts/plot_noise.py. Run: ./plot_noise.py -h for help.

Overview of calculations:
    Voc = E * Leff

    # S-parameters are measured using virtual network analyzer (VNA). A-parameters are computed from S-parameters.
    [A]    = [LNA]*[Balun A]*[Cable]*[VGA+Filter]     # total RF chain A-parameters without balun2
    Z_load = 50 * (1 + S11)/(1 - S11)                 # S11 for balun after VGA+Filter measured using VNA.
    Z_in   = (A11*Z_load + A12) / (A21*Z_load + A22)
    [A]    = [A] * [A balun2]                        # total RF chain A-parameters
    Z_ant  = antenna impedance computed from simulation

    # current and voltage at input of Balun
    I_in_BA = Voc / (Z_ant + Z_in)
    V_in_BA = I_in_BA * Z_in

    # Final voltage output at AD Chip in frequency domain.
    V_out = A11*V_in_BA + A12*I_in_BA
    I_out = A21*V_in_BA + A22*I_in_BA
"""

def read_config(xml_file):
    """ Reads the XML configuration file and returns component settings. """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    components = {}
    for comp in root.find("Components"):
        name = comp.attrib["name"]
        s2p_file = comp.find("s2pFile").text if comp.find("s2pFile") is not None else None
        s1p_file = comp.find("s1pFile").text if comp.find("s1pFile") is not None else None
        enabled = comp.find("enabled").text.lower() == "true"

        components[name] = {"s2p_file": s2p_file, "s1p_file": s1p_file, "enabled": enabled}

    csv_files = {}
    for csv in root.find("CSVFiles"):
        name = csv.attrib["name"]
        csv_file = csv.find("csvFile").text
        enabled = csv.find("enabled").text.lower() == "true"

        csv_files[name] = {"csv_file": csv_file, "enabled": enabled}

    return components, csv_files

# Load XML configuration
# xml_file = "/home/grand/grand/sim/detector/rf_chain_config.xml"  # Ensure absolute path
xml_file = Path(__file__).parent / "rf_chain_config.xml"  # Ensure absolute path
components, csv_files = read_config(xml_file)

# Dictionary to map components that depend on axis
axis_dict = {0: "X", 1: "Y", 2: "Z", "X": "X", "Y": "Y", "Z": "Z"}  # Adjust if necessary

# Function to get filenames dynamically based on axis
def get_axis_filename(component_name, axis):
    """Returns the correct filename for a given component and axis."""

    # Define a dictionary that maps numerical and string axes correctly
    axis_dict = {0: "X", 1: "Y", 2: "Z", "X": "X", "Y": "Y", "Z": "Z"}

    # Ensure axis is properly converted if it is a string digit
    if isinstance(axis, str) and axis.isdigit():
        axis = int(axis)  # Convert "0", "1", "2" to integers

    if component_name in components:
        if not components[component_name]["enabled"]:
            print(f"Warning: {component_name} is disabled in rf_chain_config.xml.")
            return None

        filename_template = components[component_name]["s2p_file"]

        if filename_template is None:
            print(f"ERROR: No filename template found for {component_name} in rf_chain_config.xml.")
            return None

        # Ensure axis replacement works correctly
        if "{axis}" in filename_template:
            if axis in axis_dict:
                resolved_filename = filename_template.replace("{axis}", axis_dict[axis])
                #print(f"DEBUG: Resolved filename for {component_name}: {resolved_filename}")
                return resolved_filename
            else:
                #print(f"ERROR: Invalid axis '{axis}' for {component_name}. Must be 0, 1, 2, X, Y, or Z.")
                return None

        #print(f"DEBUG: Using static filename for {component_name}: {filename_template}")
        return filename_template  # If no {axis} placeholder, return as is.

    print(f"ERROR: {component_name} is missing from rf_chain_config.xml.")
    return None

# Function to safely set the filename for MatchingNetwork
def _set_name_data_file(self, axis):
    r"""Returns the path of the data file for one antenna arm.

        Parameters
        ----------
        axis : int
            Arm index: 0 for X, 1 for Y, 2 for Z.

        Returns
        -------
        str
            Path to the tabulated measurements for that arm.
    """
    filename = get_axis_filename("MatchingNetwork", axis)

    #print(f"DEBUG: Final MatchingNetwork filename for {axis}: {filename}")

    if filename is None:
        raise FileNotFoundError(f"ERROR: No valid file found for MatchingNetwork with axis {axis}. Check rf_chain_config.xml.")

    return grand_add_path_data(filename)


def read_config(xml_file):
    """ Reads the XML configuration file and returns component settings. """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    components = {}
    for comp in root.find("Components"):
        name = comp.attrib["name"]
        s2p_file = comp.find("s2pFile").text if comp.find("s2pFile") is not None else None
        s1p_file = comp.find("s1pFile").text if comp.find("s1pFile") is not None else None
        enabled = comp.find("enabled").text.lower() == "true"

        components[name] = {"s2p_file": s2p_file, "s1p_file": s1p_file, "enabled": enabled}

    csv_files = {}
    for csv in root.find("CSVFiles"):
        name = csv.attrib["name"]
        csv_file = csv.find("csvFile").text
        enabled = csv.find("enabled").text.lower() == "true"

        csv_files[name] = {"csv_file": csv_file, "enabled": enabled}

    return components, csv_files

# Load XML configuration
#xml_file = "rf_chain_config.xml"
#xml_file = "/home/grand/grand/grand/sim/detector/rf_chain_config.xml"
# xml_file = "/home/grand/grand/sim/detector/rf_chain_config.xml"
xml_file = Path(__file__).parent / "rf_chain_config.xml"
components, csv_files = read_config(xml_file)

# Dictionary to map components that depend on axis
axis_dict = {0: "X", 1: "Y", 2: "Z", "X": "X", "Y": "Y", "Z": "Z"}  # Adjust if necessary

# Function to get filenames dynamically based on axis
#def get_axis_filename(component_name, axis):
#    if component_name in components and components[component_name]["enabled"]:
#        filename_template = components[component_name]["s2p_file"]
#        if filename_template and "{axis}" in filename_template:
#            return filename_template.replace("{axis}", axis_dict[axis])
#        return filename_template
#    return None

def get_axis_filename(component_name, axis):
    """Returns the correct filename for a given component and axis."""
    if component_name in components:
        if not components[component_name]["enabled"]:
            print(f"Warning: {component_name} is disabled in rf_chain_config.xml.")
            return None

        filename_template = components[component_name]["s2p_file"]

        if filename_template is None:
            print(f"ERROR: No filename template found for {component_name} in rf_chain_config.xml.")
            return None

        # Ensure axis is valid before replacing it
        if "{axis}" in filename_template:
            if axis in axis_dict:
                resolved_filename = filename_template.replace("{axis}", axis_dict[axis])
                #print(f"DEBUG: Resolved filename for {component_name}: {resolved_filename}")
                return resolved_filename
            else:
                print(f"ERROR: Invalid axis '{axis}' for {component_name}.")
                return None

        #print(f"DEBUG: Using static filename for {component_name}: {filename_template}")
        return filename_template  # If no {axis} placeholder, return as is.

    print(f"ERROR: {component_name} is missing from rf_chain_config.xml.")
    return Nonec

def interp(x,y,z):
    r"""Returns `z` interpolated onto `x` from samples at `y`.

        A thin wrapper over :func:`numpy.interp`, kept so that the interpolation
        used across this module can be changed in one place.

        Parameters
        ----------
        x : ndarray
            Positions to interpolate onto.
        y : ndarray
            Sample positions.
        z : ndarray
            Sample values.

        Returns
        -------
        ndarray
            Interpolated values, of the shape of `x`.
    """
    return np.interp(x,y,z)

def interpol_at_new_x(a_x, a_y, new_x):
    r"""Returns `a_y` interpolated onto `new_x`, zero outside the input range.

    Parameters
    ----------
    a_x : ndarray, shape (n,)
        Sample positions, in increasing order.
    a_y : ndarray, shape (n,)
        Values at `a_x`.
    new_x : ndarray, shape (m,)
        Positions to interpolate onto.

    Returns
    -------
    ndarray, shape (m,)
        Interpolated values.  Cubic within the range of `a_x`, and **zero**
        outside it rather than extrapolated -- the S-parameter tables are
        measured over 30-250 MHz and have no meaning beyond it.
    """
    assert a_x.shape[0] > 0
    #func_interpol = interpolate.interp1d(
    #    a_x, a_y, "cubic", bounds_error=False, fill_value=(1.0, 1.0)
    #)
    #return func_interpol(new_x)
    return np.interp(new_x, a_x, a_y)

def db2reim(dB, phase):
    r"""Returns the real and imaginary parts of a quantity given in decibels.

    A magnitude in decibels and a phase describe a complex number in polar
    form.  This converts that pair to Cartesian form, using the voltage
    convention :math:`|z| = 10^{dB/20}` rather than the power convention
    :math:`10^{dB/10}`, because the S-parameters this module reads are
    measured as voltage ratios by a vector network analyser.

    Parameters
    ----------
    dB : array_like
        Magnitude in decibels.
    phase : array_like
        Phase in **radians**.  Note the unit: the tabulated S-parameter
        files store degrees, so callers convert with :func:`numpy.deg2rad`
        before calling this.

    Returns
    -------
    re : ndarray
        Real part.
    im : ndarray
        Imaginary part.

    Examples
    --------
    A 0 dB magnitude is unit modulus, so a quarter-turn of phase lands on
    the imaginary axis:

    .. jupyter-execute::

        import numpy as np
        from grand.sim.detector.rf_chain import db2reim

        re, im = db2reim(np.array([0.0, -20.0]), np.array([0.0, np.pi/2]))
        print("re =", np.round(re, 6))
        print("im =", np.round(im, 6))

    The second entry shows the voltage convention: -20 dB is a factor of
    ten in amplitude, not a factor of a hundred.
    """
    mag = 10 ** (dB / 20)

    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    
    return re, im

def s2abcd(s11, s21, s12, s22):
    r"""Returns the normalised ABCD matrix of a two-port from its S-parameters.

    Scattering parameters are what a vector network analyser measures, but
    they do not cascade: the S-matrix of two networks in series is not the
    product of their S-matrices.  The ABCD (transmission) representation
    does cascade, which is the whole reason for this conversion — it is what
    lets :class:`RFChain` obtain the response of the complete chain by
    multiplying the matrices of the LNA, the baluns, the cable and the
    VGA-plus-filter in order.

    The normalised form returned here assumes equal reference impedances at
    both ports, which holds for the 50 :math:`\Omega` measurements this
    module reads.

    Parameters
    ----------
    s11, s21, s12, s22 : array_like
        The four complex scattering parameters, each of shape ``(n_freq,)``.
        Obtain them from tabulated magnitude and phase with
        :func:`db2reim`.

    Returns
    -------
    ndarray
        The ABCD matrix, shape ``(2, 2, n_freq)``, whose entries are

        .. math::

           A = \frac{(1+S_{11})(1-S_{22}) + S_{12}S_{21}}{2S_{21}},\quad
           B = \frac{(1+S_{11})(1+S_{22}) - S_{12}S_{21}}{2S_{21}},

           C = \frac{(1-S_{11})(1-S_{22}) - S_{12}S_{21}}{2S_{21}},\quad
           D = \frac{(1-S_{11})(1+S_{22}) + S_{12}S_{21}}{2S_{21}}.

    Notes
    -----
    A matched, lossless through-line has :math:`S_{11}=S_{22}=0` and
    :math:`S_{12}=S_{21}=1`, for which the ABCD matrix is the identity —
    the check in the example below, and a useful invariant for any test of
    this function.

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        from grand.sim.detector.rf_chain import s2abcd

        one  = np.ones(3, dtype=complex)
        zero = np.zeros(3, dtype=complex)
        abcd = s2abcd(zero, one, one, zero)          # ideal through-line

        print("shape:", abcd.shape)
        print("identity at every frequency:",
              np.allclose(abcd, np.eye(2)[:, :, None]))

    See Also
    --------
    matmul : cascades two ABCD matrices.
    RFChain.compute_for_freqs : builds the full chain from its stages.

    References
    ----------
    Section 8.3 of `arXiv:2408.10926 <https://arxiv.org/abs/2408.10926>`_,
    and the Xidian University RF-chain note cited in the module docstring.
    """
    return np.asarray([
        [((1+s11)*(1-s22) + s12*s21)/(2*s21), ((1+s11)*(1+s22) - s12*s21)/(2*s21)],
        [((1-s11)*(1-s22) - s12*s21)/(2*s21), ((1-s11)*(1+s22) + s12*s21)/(2*s21)]
        ])

def matmul(A, B):
    r"""Returns the product of two stacked 2x2 matrices.

    Multiplies elementwise over the trailing axes, which is what cascades
    two stages of the RF chain: an ABCD matrix per port and per frequency.

    Parameters
    ----------
    A, B : ndarray, shape (2, 2, n_ports, n_freqs)
        The matrices to multiply.

    Returns
    -------
    ndarray, shape (2, 2, n_ports, n_freqs)
        The product, computed as

        .. math::

           AB = \begin{bmatrix}
                A_{11}B_{11} + A_{12}B_{21} & A_{11}B_{12} + A_{12}B_{22} \\
                A_{21}B_{11} + A_{22}B_{21} & A_{21}B_{12} + A_{22}B_{22}
                \end{bmatrix}

    See Also
    --------
    s2abcd : produces the matrices this cascades.
    """
    assert A.shape[0]==2
    assert A.shape[1]==2
    assert A.shape[1]==B.shape[0]

    return np.asarray([
        [A[0,0]*B[0,0] + A[0,1]*B[1,0], A[0,0]*B[0,1] + A[0,1]*B[1,1]],
        [A[1,0]*B[0,0] + A[1,1]*B[1,0], A[1,0]*B[0,1] + A[1,1]*B[1,1]]
        ])


class GenericProcessingDU:
    """
    Define common attribut for frequencies for all DU effects processing
    """

    def __init__(self):
        r"""Initialises the empty arrays every chain stage shares.

                Subclasses fill them in :meth:`compute_for_freqs`.
        """
        """ """
        self.freqs_mhz = np.zeros(0)
        self.nb_freqs = 0
        self.size_sig = 0

    def _set_name_data_file(self, axis):
        """

        :param axis:
        """
        # fix a file version for processing by heritage
        pass

    ### SETTER

    def set_out_freq_mhz(self, freqs_mhz):
        """Define frequencies

        :param freqs_mhz: [MHz] given by scipy.fft.rfftfreq/1e6
        :type freqs_mhz: float (nb_freqs)
        """
        assert isinstance(freqs_mhz, np.ndarray)
        self.freqs_mhz = freqs_mhz
        self.nb_freqs = freqs_mhz.shape[0]
        self.size_sig = (self.nb_freqs - 1) * 2

class MatchingNetwork(GenericProcessingDU):
    

    r"""The impedance matching network between antenna and LNA.
    """
    def __init__(self):
        """

        :param size_sig: size of the trace after
        """
        super().__init__()
        #self.data_lna = []
        self.sparams = []
        for axis in range(3):
            matcnet = np.loadtxt(self._set_name_data_file(axis), comments=['#', '!'])
            self.sparams.append(matcnet)
        self.freqs_in = matcnet[:, 0] / 1e6   # note: freqs_in for x and y ports is the same, but for z port is different.
        self.nb_freqs_in = len(self.freqs_in)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis):
        """

        ! Created Wed May 10 01:24:03 2023
        # hz S ma R 50
        ! 2 Port Network Data from SP1.SP block
        """
        axis_dict = {0:"X", 1:"Y", 2:"Z"}
        filename = get_axis_filename("MatchingNetwork", axis)

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        
        r"""Computes this stage's response on the given frequency axis.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            Output frequency axis, in MHz.  The tabulated S-parameters are
            interpolated onto it, and are zero outside the 30-250 MHz band they
            were measured over.

        Notes
        -----
        Results are stored on the instance rather than returned; the chain reads
        them when it cascades the stages.
        """
        logger.debug(f"{self.sparams[0].shape}")
        self.set_out_freq_mhz(freqs_mhz)
        assert self.nb_freqs > 0

        # nb_freqs in __init__ is 0. nb_freqs changes after self.set_out_freq_mhz(freqs_mhz)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements dB, phase[deg]: S11, S21, S12, S22
        # Fill S-parameters from files obtained by measuring S-parameters using virtual network analyzer.
        for axis in range(3):
            freqs_in = self.sparams[axis][:, 0] / 1e6 # note: freqs_in for x and y ports is the same, but for z port is different.
            # ----- S11
            dbs11 = self.sparams[axis][:, 1]
            phs11 = np.deg2rad(self.sparams[axis][:, 2])
            #res11, ims11 = db2reim(dbs11, phs11)
            res11 = dbs11 * np.cos(phs11)
            ims11 = dbs11 * np.sin(phs11)
            self.dbs11[axis] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S21
            dbs21 = self.sparams[axis][:, 3]
            phs21 = np.deg2rad(self.sparams[axis][:, 4])
            #res21, ims21 = db2reim(dbs21, phs21)
            res21 = dbs21 * np.cos(phs21)
            ims21 = dbs21 * np.sin(phs21)
            self.dbs21[axis] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S12
            dbs12 = self.sparams[axis][:, 5]
            phs12 = np.deg2rad(self.sparams[axis][:, 6])
            #res12, ims12 = db2reim(dbs12, phs12)
            res12 = dbs12 * np.cos(phs12)
            ims12 = dbs12 * np.sin(phs12)
            self.dbs12[axis] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S22
            dbs22 = self.sparams[axis][:, 7]
            phs22 = np.deg2rad(self.sparams[axis][:, 8])
            #res22, ims22 = db2reim(dbs22, phs22)
            res22 = dbs22 * np.cos(phs22)
            ims22 = dbs22 * np.sin(phs22)
            self.dbs22[axis] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)
        #xy_denorm_factor = np.array([[1, 100], [1/100., 1]]) # denormalizing factor for XY arms
        #xy_denorm_factor = np.array([[1, 100], [1/100., 1]]) # denormalizing factor for XY arms
        xy_denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XY arms
        xy_denorm_factor = xy_denorm_factor[..., np.newaxis, np.newaxis]
        #z_denorm_factor = np.array([[1, 50], [1/50., 1]])    # denormalizing factor for Z arms
        z_denorm_factor = np.array([[1, 50], [1/50., 1]])    # denormalizing factor for Z arms
        z_denorm_factor = z_denorm_factor[..., np.newaxis]

        ABCD_matrix = s2abcd(self.s11, self.s21, self.s12, self.s22) # this is a normalized A-matrix represented by [a] in the document.

        ABCD_matrix[..., :2, :] *= xy_denorm_factor # denormalizing factor for XY arms
        ABCD_matrix[..., 2, :] *= z_denorm_factor   # denormalizing factor for Z arm
        self.ABCD_matrix[:] = ABCD_matrix # this is an A-matrix represented by [A] in the document.


class gaa_frontend0db(GenericProcessingDU):
    

    r"""The GAA front end at 0 dB gain.
    """
    def __init__(self):
        """

        :param size_sig: size of the trace after
        """
        super().__init__()
        #self.data_lna = []
        self.sparams = []
        for axis in range(3):
            matcnet = np.loadtxt(self._set_name_data_file(axis), comments=['#', '!'])
            self.sparams.append(matcnet)
        self.freqs_in = matcnet[:, 0] / 1e6   # note: freqs_in for x and y ports is the same, but for z port is different.
        #self.freqs_in = matcnet[:, 0]   # note: freqs_in for x and y ports is the same, but for z port is different.
        self.nb_freqs_in = len(self.freqs_in)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis):
        """

        ! Created Wed May 10 01:24:03 2023
        # hz S ma R 50
        ! 2 Port Network Data from SP1.SP block
        """
        axis_dict = {0:"X", 1:"Y", 2:"Z"}
        filename = get_axis_filename("AntennaLNA", axis)
        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        
        r"""Computes this stage's response on the given frequency axis.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            Output frequency axis, in MHz.  The tabulated S-parameters are
            interpolated onto it, and are zero outside the 30-250 MHz band they
            were measured over.

        Notes
        -----
        Results are stored on the instance rather than returned; the chain reads
        them when it cascades the stages.
        """
        logger.debug(f"{self.sparams[0].shape}")
        self.set_out_freq_mhz(freqs_mhz)
        assert self.nb_freqs > 0

        # nb_freqs in __init__ is 0. nb_freqs changes after self.set_out_freq_mhz(freqs_mhz)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements dB, phase[deg]: S11, S21, S12, S22
        # Fill S-parameters from files obtained by measuring S-parameters using virtual network analyzer.
        for axis in range(3):
            freqs_in = self.sparams[axis][:, 0] / 1e6 # note: freqs_in for x and y ports is the same, but for z port is different.
            # ----- S11
            dbs11 = self.sparams[axis][:, 1]
            phs11 = np.deg2rad(self.sparams[axis][:, 2])
            res11, ims11 = db2reim(dbs11, phs11)
            #res11 = dbs11 * np.cos(phs11)
            #ims11 = dbs11 * np.sin(phs11)
            self.dbs11[axis] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S21
            dbs21 = self.sparams[axis][:, 3]
            phs21 = np.deg2rad(self.sparams[axis][:, 4])
            res21, ims21 = db2reim(dbs21, phs21)
            #res21 = dbs21 * np.cos(phs21)
            #ims21 = dbs21 * np.sin(phs21)
            self.dbs21[axis] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S12
            dbs12 = self.sparams[axis][:, 5]
            phs12 = np.deg2rad(self.sparams[axis][:, 6])
            res12, ims12 = db2reim(dbs12, phs12)
            #res12 = dbs12 * np.cos(phs12)
            #ims12 = dbs12 * np.sin(phs12)
            self.dbs12[axis] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S22
            dbs22 = self.sparams[axis][:, 7]
            phs22 = np.deg2rad(self.sparams[axis][:, 8])
            res22, ims22 = db2reim(dbs22, phs22)
            #res22 = dbs22 * np.cos(phs22)
            #ims22 = dbs22 * np.sin(phs22)
            self.dbs22[axis] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)
        #xy_denorm_factor = np.array([[1, 100], [1/100., 1]]) # denormalizing factor for XY arms
        #xy_denorm_factor = np.array([[1, 100], [1/100., 1]]) # denormalizing factor for XY arms
        xy_denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XY arms
        xy_denorm_factor = xy_denorm_factor[..., np.newaxis, np.newaxis]
        #z_denorm_factor = np.array([[1, 50], [1/50., 1]])    # denormalizing factor for Z arms
        z_denorm_factor = np.array([[1, 50], [1/50., 1]])    # denormalizing factor for Z arms
        z_denorm_factor = z_denorm_factor[..., np.newaxis]

        ABCD_matrix = s2abcd(self.s11, self.s21, self.s12, self.s22) # this is a normalized A-matrix represented by [a] in the document.

        ABCD_matrix[..., :2, :] *= xy_denorm_factor # denormalizing factor for XY arms
        ABCD_matrix[..., 2, :] *= z_denorm_factor   # denormalizing factor for Z arm
        self.ABCD_matrix[:] = ABCD_matrix # this is an A-matrix represented by [A] in the document.


        
class LowNoiseAmplifier(GenericProcessingDU):
    """

    Class goals:
      * Perform the LNA filter on signal for each antenna
      * read only once LNA data files
      * pre_compute interpolation
    """

    def __init__(self):
        """

        :param size_sig: size of the trace after
        """
        super().__init__()
        #self.data_lna = []
        self.sparams = []
        for axis in range(3):
            lna = np.loadtxt(self._set_name_data_file(axis), comments=['#', '!'])
            self.sparams.append(lna)
        self.freqs_in = lna[:, 0] / 1e6   # note: freqs_in for x and y ports is the same, but for z port is different.
        self.nb_freqs_in = len(self.freqs_in)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis):
        """

        Ceyear Technologies,3672C,ZKL00189,2.1.5
        Calibration ON : 2P/1,2
        Sweep Type: lin Frequency Sweep
        S2P File: Measurements: S11, S21, S12, S22:
        Thursday, April 27, 2023
        Hz  S  dB  R 50.000
        """
        axis_dict = {0:"X", 1:"Y", 2:"Z"}
        filename = get_axis_filename("LNA", axis)

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """
        compute s-parameters of LNA

        """
        logger.debug(f"{self.sparams[0].shape}")
        self.set_out_freq_mhz(freqs_mhz)
        assert self.nb_freqs > 0

        # nb_freqs in __init__ is 0. nb_freqs changes after self.set_out_freq_mhz(freqs_mhz)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements dB, phase[deg]: S11, S21, S12, S22
        # Fill S-parameters from files obtained by measuring S-parameters using virtual network analyzer.
        for axis in range(3):
            freqs_in = self.sparams[axis][:, 0] / 1e6 # note: freqs_in for x and y ports is the same, but for z port is different.
            # ----- S11
            dbs11 = self.sparams[axis][:, 1]
            phs11 = np.deg2rad(self.sparams[axis][:, 2])
            res11, ims11 = db2reim(dbs11, phs11)
            self.dbs11[axis] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S21
            dbs21 = self.sparams[axis][:, 3]
            phs21 = np.deg2rad(self.sparams[axis][:, 4])
            res21, ims21 = db2reim(dbs21, phs21)
            self.dbs21[axis] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S12
            dbs12 = self.sparams[axis][:, 5]
            phs12 = np.deg2rad(self.sparams[axis][:, 6])
            res12, ims12 = db2reim(dbs12, phs12)
            self.dbs12[axis] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S22
            dbs22 = self.sparams[axis][:, 7]
            phs22 = np.deg2rad(self.sparams[axis][:, 8])
            res22, ims22 = db2reim(dbs22, phs22)
            self.dbs22[axis] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)
        xy_denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XY arms
        #xy_denorm_factor = np.array([[1, 100], [1/100., 1]]) # denormalizing factor for XY arms
        xy_denorm_factor = xy_denorm_factor[..., np.newaxis, np.newaxis]
        z_denorm_factor = np.array([[1, 50], [1/50., 1]])    # denormalizing factor for Z arms
        z_denorm_factor = z_denorm_factor[..., np.newaxis]

        ABCD_matrix = s2abcd(self.s11, self.s21, self.s12, self.s22) # this is a normalized A-matrix represented by [a] in the document.

        ABCD_matrix[..., :2, :] *= xy_denorm_factor # denormalizing factor for XY arms
        ABCD_matrix[..., 2, :] *= z_denorm_factor   # denormalizing factor for Z arm
        self.ABCD_matrix[:] = ABCD_matrix # this is an A-matrix represented by [A] in the document.

class BalunAfterLNA(GenericProcessingDU):
    """
    Class goals:
      * deals with Balun after LNA (inside Nut). 
      * Note that balun is placed after LNA in version 1. The same type of balun is placed before matching-network and LNA in version 2.
      * Balun is used in X and Y ports only. No Balun in Z port.
      * Balun without matching network is used in version 1.
    """

    def __init__(self):
        r"""Initialises the balun that follows the low-noise amplifier.
        """
        """ """
        super().__init__()
        #self.data_cable = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz
        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis=0):
        """

        Created: May 4, 2023
        hz S ma R 50
        2 Port Network Data from SP1.SP block
        freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22         
        """
        #filename = os.path.join("detector", "RFchain_v1", "balun_after_LNA.s2p")
        #filename = os.path.join("detector", "RFchain_v1", "balun46in.s2p")
        filename = components["BalunIn"]["s2p_file"] if components["BalunIn"]["enabled"] else None
        
        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """Compute ABCD_matrix for frequency freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64) # shape = (2x2 matrix, 3 ports, nb_freqs)

        # freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22
        # ----- S11
        mags11 = self.sparams[:, 1]
        angs11 = np.deg2rad(self.sparams[:, 2])
        res11 = mags11 * np.cos(angs11)
        ims11 = mags11 * np.sin(angs11)
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S21
        mags21 = self.sparams[:, 3]
        angs21 = np.deg2rad(self.sparams[:, 4])
        res21 = mags21 * np.cos(angs21)
        ims21 = mags21 * np.sin(angs21)
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S12
        mags12 = self.sparams[:, 5]
        angs12 = np.deg2rad(self.sparams[:, 6])
        res12 = mags12 * np.cos(angs12)
        ims12 = mags12 * np.sin(angs12)
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S22
        mags22 = self.sparams[:, 7]
        angs22 = np.deg2rad(self.sparams[:, 8])
        res22 = mags22 * np.cos(angs22)
        ims22 = mags22 * np.sin(angs22)
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis] # match with the shape of ABCD_matrix to broadcast.
        # for X and Y ports only. No Balun in Z port. shape of ABCD_matrix is (2, 2, 3, nb_freqs).     
        self.ABCD_matrix[:] = s2abcd(self.s11, self.s21, self.s12, self.s22) * denorm_factor
        # force components of ABCD_matrix for Z port to be identity matrix because there is no Balun in Z port.
        #self.ABCD_matrix[:,:,2,:] = np.identity(2)[...,np.newaxis]  # add np.newaxis to broadcast to all frequencies.

class Cable(GenericProcessingDU):
    """

    Class goals:
      * pre_compute interpolation
    """

    def __init__(self):
        r"""Initialises the cable and its connector.
        """
        """ """
        super().__init__()
        #self.data_cable = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz

        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis=0):
        """

        :param axis:
        """
        filename = components["CableConnector"]["s2p_file"] if components["CableConnector"]["enabled"] else None

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """Compute ABCD_matrix for frequency freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements: S11, S21, S12, S22
        # ----- S11
        dbs11 = self.sparams[:, 1]
        phs11 = np.deg2rad(self.sparams[:, 2])
        res11, ims11 = db2reim(dbs11, phs11)
        self.dbs11[:] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S21
        dbs21 = self.sparams[:, 3]
        phs21 = np.deg2rad(self.sparams[:, 4])
        res21, ims21 = db2reim(dbs21, phs21)
        self.dbs21[:] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S12
        dbs12 = self.sparams[:, 5]
        phs12 = np.deg2rad(self.sparams[:, 6])
        res12, ims12 = db2reim(dbs12, phs12)
        self.dbs12[:] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S22
        dbs22 = self.sparams[:][:, 7]
        phs22 = np.deg2rad(self.sparams[:, 8])
        res22, ims22 = db2reim(dbs22, phs22)
        self.dbs22[:] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis] # match with the shape of ABCD_matrix.
        # for all three ports. shape of ABCD_matrix is (2, 2, ant ports, nb_freqs)   .     
        self.ABCD_matrix[:] = s2abcd(self.s11, self.s21, self.s12, self.s22) * denorm_factor

class VGAFilter(GenericProcessingDU):
    """

    Class goals:
      * pre_compute interpolation
    """

    def __init__(self, gain=0):
        """ 
        :param gain: gain setup for VGA in dB.
        """
        super().__init__()

        self.gain = gain
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz

        # shape = (nports, nfreqs). self.nb_freqs here is 0.
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis=0):
        """

        :param axis:
        """
        assert self.gain in [-5, 0, 5, 20]
        logger.info(f"vga gain: {self.gain} dB")
        #filename = os.path.join("detector", "RFchain_v2", "filter+"f"vga{self.gain}db+filter.s2p")
        filename = components["Filter"]["s2p_file"] if components["Filter"]["enabled"] else None
        
        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """Compute ABCD_matrix for frequency freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements: S11, S21, S12, S22
        # ----- S11
        dbs11 = self.sparams[:, 1]
        phs11 = np.deg2rad(self.sparams[:, 2])
        res11, ims11 = db2reim(dbs11, phs11)
        self.dbs11[:] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S21
        dbs21 = self.sparams[:, 3]
        phs21 = np.deg2rad(self.sparams[:, 4])
        res21, ims21 = db2reim(dbs21, phs21)
        self.dbs21[:] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S12
        dbs12 = self.sparams[:, 5]
        phs12 = np.deg2rad(self.sparams[:, 6])
        res12, ims12 = db2reim(dbs12, phs12)
        self.dbs12[:] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S22
        dbs22 = self.sparams[:][:, 7]
        phs22 = np.deg2rad(self.sparams[:, 8])
        res22, ims22 = db2reim(dbs22, phs22)
        self.dbs22[:] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis]
        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)        
        ABCD_matrix = s2abcd(self.s11, self.s21, self.s12, self.s22)
        ABCD_matrix *= denorm_factor # denormalizing factor for XYZ arms
        self.ABCD_matrix[:] = ABCD_matrix

class BalunBeforeADC(GenericProcessingDU):
    """Class goals:
      * Pass signal through Balun before Analog to Digitial Converter (ADC) for each antenna
      * Balun is used in x, y, and z ports
      * Same data is used for all three ports
      * read data files only once
      * pre_compute interpolation
      * this Balun is referred to as Balun1 Balun2
    """

    def __init__(self):
        """ 
        :param sparams: S-parameters data for x, y, and z ports. Same data is used for x, y, and z ports.
        :param freqs_in: frequencies corresponding to the S-parameters data for x, y, and z ports.
        :param s11, s21, s12, s22: S-parameters for x, y, and z ports. shape (3, nb_freqs).
        :param ABCD_matrix: not normalized ABCD matrix corresponding to S-parameters. shape (2, 2, nb_ports, nb_freqs)
        """
        super().__init__()
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz
        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self):
        """Created Mon May 15, 2023 11:21:18 2023
        hz S ma R 50
        2 Port Network Data from SP1.SP block
        freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22  
        """
        filename = components["BalunBeforeAD"]["s2p_file"] if components["BalunBeforeAD"]["enabled"] else None

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """compute s-parameters and ABCD matrix of Balun before AD chip for freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64) # shape = (2x2 matrix, 3 ports, nb_freqs)

        # freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22
        # ----- S11
        mags11 = self.sparams[:, 1]
        angs11 = np.deg2rad(self.sparams[:, 2])
        res11 = mags11 * np.cos(angs11)
        ims11 = mags11 * np.sin(angs11)
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S21
        mags21 = self.sparams[:, 3]
        angs21 = np.deg2rad(self.sparams[:, 4])
        res21 = mags21 * np.cos(angs21)
        ims21 = mags21 * np.sin(angs21)
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S12
        mags12 = self.sparams[:, 5]
        angs12 = np.deg2rad(self.sparams[:, 6])
        res12 = mags12 * np.cos(angs12)
        ims12 = mags12 * np.sin(angs12)
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S22
        mags22 = self.sparams[:, 7]
        angs22 = np.deg2rad(self.sparams[:, 8])
        res22 = mags22 * np.cos(angs22)
        ims22 = mags22 * np.sin(angs22)
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)
        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis]
        self.ABCD_matrix[:] = s2abcd(self.s11, self.s21, self.s12, self.s22) * denorm_factor # this is an A-matrix represented by [A] in the document.

#################################################################################################
        
class Rfchain_elements_db(GenericProcessingDU):
    r"""A chain element whose response is tabulated in decibels.
    """
    def __init__(self, filename="test2.s2p"):
        r"""Loads this stage's tabulated data and prepares its S-parameters.

        Parameters
        ----------
        filename : str
            Path to the measured data file for this element.
        """
        super().__init__()
        self.filename = filename

        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self):
        r"""Returns the path of the data file holding this stage's measurements.
        """
        filename = os.path.join("detector", "RFchain_v2", self.filename)
        return grand_add_path_data(filename)
    def compute_for_freqs(self, freqs_mhz):
        r"""Computes this stage's response on the given frequency axis.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            Output frequency axis, in MHz.  The tabulated S-parameters are
            interpolated onto it, and are zero outside the 30-250 MHz band they
            were measured over.

        Notes
        -----
        Results are stored on the instance rather than returned; the chain reads
        them when it cascades the stages.
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements: S11, S21, S12, S22
        # ----- S11
        dbs11 = self.sparams[:, 1]
        phs11 = np.deg2rad(self.sparams[:, 2])
        res11, ims11 = db2reim(dbs11, phs11)
        self.dbs11[:] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) 
        # ----- S21
        dbs21 = self.sparams[:, 3]
        phs21 = np.deg2rad(self.sparams[:, 4])
        res21, ims21 = db2reim(dbs21, phs21)
        self.dbs21[:] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) 
        # ----- S12
        dbs12 = self.sparams[:, 5]
        phs12 = np.deg2rad(self.sparams[:, 6])
        res12, ims12 = db2reim(dbs12, phs12)
        self.dbs12[:] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) 
        # ----- S22
        dbs22 = self.sparams[:][:, 7]
        phs22 = np.deg2rad(self.sparams[:, 8])
        res22, ims22 = db2reim(dbs22, phs22)
        self.dbs22[:] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) 

        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis]
        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)        
        ABCD_matrix = s2abcd(self.s11, self.s21, self.s12, self.s22)
        ABCD_matrix *= denorm_factor # denormalizing factor for XYZ arms
        self.ABCD_matrix[:] = ABCD_matrix

########################################################################################

class Rfchain_elements_db_rad(GenericProcessingDU):
    r"""A chain element tabulated in decibels, with phase in radians.
    """
    def __init__(self, filename="test2.s2p"):
        r"""Loads this stage's tabulated data and prepares its S-parameters.

        Parameters
        ----------
        filename : str
            Path to the measured data file for this element.
        """
        super().__init__()
        self.filename = filename

        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self):
        r"""Returns the path of the data file holding this stage's measurements.
        """
        filename = os.path.join("detector", "RFchain_v2", self.filename)
        return grand_add_path_data(filename)
    def compute_for_freqs(self, freqs_mhz):
        r"""Computes this stage's response on the given frequency axis.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            Output frequency axis, in MHz.  The tabulated S-parameters are
            interpolated onto it, and are zero outside the 30-250 MHz band they
            were measured over.

        Notes
        -----
        Results are stored on the instance rather than returned; the chain reads
        them when it cascades the stages.
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements: S11, S21, S12, S22
        # ----- S11
        dbs11 = self.sparams[:, 1]
        phs11 = self.sparams[:, 2]
        res11, ims11 = db2reim(dbs11, phs11)
        self.dbs11[:] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) 
        # ----- S21
        dbs21 = self.sparams[:, 3]
        phs21 = self.sparams[:, 4]
        res21, ims21 = db2reim(dbs21, phs21)
        self.dbs21[:] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) 
        # ----- S12
        dbs12 = self.sparams[:, 5]
        phs12 = self.sparams[:, 6]
        res12, ims12 = db2reim(dbs12, phs12)
        self.dbs12[:] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) 
        # ----- S22
        dbs22 = self.sparams[:][:, 7]
        phs22 = self.sparams[:, 8]
        res22, ims22 = db2reim(dbs22, phs22)
        self.dbs22[:] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) 

        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis]
        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)        
        ABCD_matrix = s2abcd(self.s11, self.s21, self.s12, self.s22)
        ABCD_matrix *= denorm_factor # denormalizing factor for XYZ arms
        self.ABCD_matrix[:] = ABCD_matrix
        
###################################################################################### 

class Rfchain_elements(GenericProcessingDU):
    r"""A chain element whose response is tabulated in linear units.
    """
    def __init__(self, filename="test.s2p"):
        r"""Loads this stage's tabulated data and prepares its S-parameters.

        Parameters
        ----------
        filename : str
            Path to the measured data file for this element.
        """
        super().__init__()
        self.filename = filename
        
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz
        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self):
        r"""Returns the path of the data file holding this stage's measurements.
        """
        filename = os.path.join("detector", "RFchain_v2", self.filename)
        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """compute s-parameters and ABCD matrix of Balun before AD chip for freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64) # shape = (2x2 matrix, 3 ports, nb_freqs)

        # freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22
        # ----- S11
        mags11 = self.sparams[:, 1]
        angs11 = np.deg2rad(self.sparams[:, 2])
        res11 = mags11 * np.cos(angs11)
        ims11 = mags11 * np.sin(angs11)
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) 
        # ----- S21
        mags21 = self.sparams[:, 3]
        angs21 = np.deg2rad(self.sparams[:, 4])
        res21 = mags21 * np.cos(angs21)
        ims21 = mags21 * np.sin(angs21)
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) 
        # ----- S12
        mags12 = self.sparams[:, 5]
        angs12 = np.deg2rad(self.sparams[:, 6])
        res12 = mags12 * np.cos(angs12)
        ims12 = mags12 * np.sin(angs12)
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) 
        # ----- S22
        mags22 = self.sparams[:, 7]
        angs22 = np.deg2rad(self.sparams[:, 8])
        res22 = mags22 * np.cos(angs22)
        ims22 = mags22 * np.sin(angs22)
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) 

        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)
        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis]
        self.ABCD_matrix[:] = s2abcd(self.s11, self.s21, self.s12, self.s22) * denorm_factor 

##############################################################################################
class Rfchain_elements_rad(GenericProcessingDU):
    r"""A chain element tabulated linearly, with phase in radians.
    """
    def __init__(self, filename="test.s2p"):
        r"""Loads this stage's tabulated data and prepares its S-parameters.

        Parameters
        ----------
        filename : str
            Path to the measured data file for this element.
        """
        super().__init__()
        self.filename = filename
        
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz
        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self):
        r"""Returns the path of the data file holding this stage's measurements.
        """
        filename = os.path.join("detector", "RFchain_v2", self.filename)
        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """compute s-parameters and ABCD matrix of Balun before AD chip for freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64) # shape = (2x2 matrix, 3 ports, nb_freqs)

        # freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22
        # ----- S11
        mags11 = self.sparams[:, 1]
        angs11 = self.sparams[:, 2]
        res11 = mags11 * np.cos(angs11)
        ims11 = mags11 * np.sin(angs11)
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) 
        # ----- S21
        mags21 = self.sparams[:, 3]
        angs21 = self.sparams[:, 4]
        res21 = mags21 * np.cos(angs21)
        ims21 = mags21 * np.sin(angs21)
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) 
        # ----- S12
        mags12 = self.sparams[:, 5]
        angs12 = self.sparams[:, 6]
        res12 = mags12 * np.cos(angs12)
        ims12 = mags12 * np.sin(angs12)
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) 
        # ----- S22
        mags22 = self.sparams[:, 7]
        angs22 = self.sparams[:, 8]
        res22 = mags22 * np.cos(angs22)
        ims22 = mags22 * np.sin(angs22)
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) 

        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)
        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis]
        self.ABCD_matrix[:] = s2abcd(self.s11, self.s21, self.s12, self.s22) * denorm_factor 

###########################################################################################

class Zload_arb(GenericProcessingDU):
    r"""An arbitrary load impedance read from a measurement file.
    """
    def __init__(self, filename="S_balun_AD.s1p"):
        r"""Loads this stage's tabulated data and prepares its S-parameters.

        Parameters
        ----------
        filename : str
            Path to the measured data file for this element.
        """
        super().__init__()
        self.filename = filename
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz
        self.s = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )
        self.Z_load = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )

    def _set_name_data_file(self):
        #filename = os.path.join("detector", "RFchain_v1", "zload_balun_200ohm.s1p")
        r"""Returns the path of the data file holding this stage's measurements.
        """
        filename = os.path.join("detector", "RFchain_v2", self.filename)
        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        r"""Computes this stage's response on the given frequency axis.

        Parameters
        ----------
        freqs_mhz : ndarray, shape (n_freq,)
            Output frequency axis, in MHz.  The tabulated S-parameters are
            interpolated onto it, and are zero outside the 30-250 MHz band they
            were measured over.

        Notes
        -----
        Results are stored on the instance rather than returned; the chain reads
        them when it cascades the stages.
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0
        self.s = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )
        self.Z_load = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )
        # S1P File: Measurements: S22
        #res = self.sparams[:, 1]
        #ims = self.sparams[:, 2]
        dbs = self.sparams[:, 1]
        phs = np.deg2rad(self.sparams[:, 2])
        res, ims = db2reim(dbs, phs)
        self.s[:] = interpol_at_new_x(freqs_in, res, self.freqs_mhz)       
        self.s[:] += 1j * interpol_at_new_x(freqs_in, ims, self.freqs_mhz) 
        # Calculation of Zload (Zload = balun+200ohm + ADchip)
        self.Z_load[:] = 50 * (1 + self.s) / (1 - self.s)


############################################################################################

class Zload(GenericProcessingDU):
    """Class goals:
      * computes input impedance of load due to balun + 200ohm ADC.
    """

    def __init__(self):
        """Reflection coefficient (self.s) is measured using VNA.
        Same value is used for all ports.
        :param sparams: S-parameters data to compute Zload for x, y, and z ports. Same Zload is used for x, y, and z ports.
        :param freqs_in: frequencies corresponding to the S-parameters data for x, y, and z ports.
        :param s: reflection coefficient for x, y, and z ports. shape (nb_freqs,).
        :param Z_load: total impedance of the load that includes balun, 200 ohm resistor and AD chip.
        """
        super().__init__()
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz
        self.s = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )
        self.Z_load = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )

    def _set_name_data_file(self, axis=0):
        """Ceyear Technologies,3672C, ZKL00189, 2.1.5
        Calibration ON : 2P/1,2
        Sweep Type: lin Frequency Sweep
        S1P File: Measurements: S22:
        Thursday, April 20, 2023
        Hz  S  RI  R 50
        """
        #filename = os.path.join("detector", "RFchain_v1", "zload_balun_200ohm.s1p")
        filename = components["S_balun_AD"]["s1p_file"] if components["S_balun_AD"]["enabled"] else None

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """compute S-paramters and Zload for freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        self.s = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )
        self.Z_load = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )

        # S1P File: Measurements: S22
        #res = self.sparams[:, 1]
        #ims = self.sparams[:, 2]
        dbs = self.sparams[:, 1]
        phs = np.deg2rad(self.sparams[:, 2])
        res, ims = db2reim(dbs, phs)
        self.s[:] = interpol_at_new_x(freqs_in, res, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s[:] += 1j * interpol_at_new_x(freqs_in, ims, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        # Calculation of Zload (Zload = balun+200ohm + ADchip)
        self.Z_load[:] = 50 * (1 + self.s) / (1 - self.s)

class RFChain(GenericProcessingDU):
    """
    Facade for all elements in RF chain
    """

    def __init__(self, vga_gain=20):
        r"""Assembles the full RF chain: matching network, LNA, baluns, cable, VGA and filter.

        Parameters
        ----------
        vga_gain : int, optional
            Gain of the variable-gain amplifier, in dB.  S-parameters are shipped
            for 20 (the GRANDProto300 default), 5, 0 and -5.

        Notes
        -----
        Construction only gathers the stages; :meth:`compute_for_freqs` evaluates
        them, and :meth:`get_tf` returns the resulting transfer function.
        """
        super().__init__()
        self.matcnet = MatchingNetwork()
        self.lna = LowNoiseAmplifier()
        self.balun1 = BalunAfterLNA()
        self.cable = Cable()
        self.vgaf = VGAFilter(gain=vga_gain)
        self.balun2 = BalunBeforeADC()
        self.zload = Zload()
        # Note: self.nb_freqs at this point is 0.
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

    def compute_for_freqs(self, freqs_mhz):
        """Compute transfer function for frequency freqs_mhz

        :param freqs_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        self.matcnet.compute_for_freqs(freqs_mhz)
        self.lna.compute_for_freqs(freqs_mhz)
        self.balun1.compute_for_freqs(freqs_mhz)
        self.cable.compute_for_freqs(freqs_mhz)
        self.vgaf.compute_for_freqs(freqs_mhz)
        self.balun2.compute_for_freqs(freqs_mhz)
        self.zload.compute_for_freqs(freqs_mhz)
        #self.balun_after_vga.compute_for_freqs(freqs_mhz)

        assert self.lna.nb_freqs > 0
        assert self.lna.ABCD_matrix.shape[-1] > 0
        assert self.lna.nb_freqs==self.balun1.nb_freqs
        
        assert self.matcnet.nb_freqs > 0
        assert self.matcnet.ABCD_matrix.shape[-1] > 0
        assert self.matcnet.nb_freqs==self.balun1.nb_freqs
        
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

        # Note that components of ABCD_matrix for Z port of balun1 is set to 1 as no Balun is used. shape = (2,2,nports,nfreqs)
        # Note that this is a matrix multiplication
        # Associative property of matrix multiplication is used, ie. (AB)C = A(BC)
        # Make sure to multiply in this order: balun1 * matching_network * lna.ABCD_matrix * cable.ABCD_matrix * vgaf.ABCD_matrix
        
        MM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        MM2 = matmul(MM1, self.lna.ABCD_matrix)
        MM3 = matmul(self.cable.ABCD_matrix, self.vgaf.ABCD_matrix)
        self.total_ABCD_matrix[:] = matmul(MM2, MM3)
        
        # Calculation of Z_in (this is the total impedence of the RF chain excluding antenna arm. see page 50 of the document.)
        self.Z_load = self.zload.Z_load[np.newaxis, :] # shape (nfreq) --> (1,nfreq) to broadcast with components of ABCD_matrix with shape (2,2,ports,nfreq).
        self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])
        #self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])
        
        # Once Z_in is calculated, calculate the final total_ABCD_matrix including Balun2.
        self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix)
        #self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix) 

        # Antenna Impedance.
        filename = csv_files["AntennaImpedance"]["csv_file"] if csv_files["AntennaImpedance"]["enabled"] else None
        filename = grand_add_path_data(filename)
        Zant_dat = np.loadtxt(filename, delimiter=",", comments=['#', '!'], skiprows=1)
        freqs_in = Zant_dat[:,0]  # MHz
        self.Z_ant[0] = interpol_at_new_x(freqs_in, Zant_dat[:,1], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[0] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,2], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] = interpol_at_new_x(freqs_in, Zant_dat[:,3], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,4], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] = interpol_at_new_x(freqs_in, Zant_dat[:,5], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,6], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.

    def vout_f(self, voc_f):
        """ Compute final voltage after propagating signal through RF chain.
        Input: Voc_f (in frequency domain)
        Output: Voltage after RF chain in frequency domain.
        Make sure to run self.compute_for_freqs() before calling this method.
        RK Note: name 'vout_f' is a placeholder. Change it with something better. 
        """
        assert voc_f.shape==self.Z_in.shape  # shape = (nports, nfreqs)

        self.I_in_balunA = voc_f / (self.Z_ant + self.Z_in)
        self.V_in_balunA = self.I_in_balunA * self.Z_in

        # loop over three ports. shape of total_ABCD_matrix is (2,2,nports,nfreqs)
        for i in range(3):
            ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_1port = np.moveaxis(ABCD_matrix_1port, -1, 0) # (2,2,nfreqs) --> (nfreqs,2,2), to compute inverse of ABCD_matrix using np.linalg.inv.
            ABCD_matrix_1port_inv = np.linalg.inv(ABCD_matrix_1port)
            V_out_RFchain = ABCD_matrix_1port_inv[:,0,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,0,1]*self.I_in_balunA[i]
            I_out_RFchain = ABCD_matrix_1port_inv[:,1,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,1,1]*self.I_in_balunA[i]

            self.V_out_RFchain[i] = V_out_RFchain
            self.I_out_RFchain[i] = I_out_RFchain

        return self.V_out_RFchain

    def get_tf(self):
        """Return transfer function for all elements in RF chain
        total transfer function is the output voltage for input Voc of 1. It says by what factor the Voc will be multiplied by the RF chain.
        @return total TF (complex, (3,N)):
        """
        self._total_tf = self.vout_f(np.ones((3, self.nb_freqs)))

        return self._total_tf

class RFChainNut(GenericProcessingDU):
    """
    Facade for all elements in RF chain
    """

    def __init__(self, vga_gain=20):
        r"""Assembles the RF chain of the Nut variant of the detection unit.

        Parameters
        ----------
        vga_gain : int, optional
            Gain of the variable-gain amplifier, in dB.  S-parameters are shipped
            for 20 (the GRANDProto300 default), 5, 0 and -5.

        Notes
        -----
        Construction only gathers the stages; :meth:`compute_for_freqs` evaluates
        them, and :meth:`get_tf` returns the resulting transfer function.
        """
        super().__init__()
        self.matcnet = MatchingNetwork()
        self.lna = LowNoiseAmplifier()
        self.balun1 = BalunAfterLNA()
        self.cable = Cable()
        self.vgaf = VGAFilter(gain=vga_gain)
        self.balun2 = BalunBeforeADC()
        self.zload = Zload()
        # Note: self.nb_freqs at this point is 0.
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

    def compute_for_freqs(self, freqs_mhz):
        """Compute transfer function for frequency freqs_mhz

        :param freqs_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        self.matcnet.compute_for_freqs(freqs_mhz)
        self.lna.compute_for_freqs(freqs_mhz)
        self.balun1.compute_for_freqs(freqs_mhz)
        self.cable.compute_for_freqs(freqs_mhz)
        self.vgaf.compute_for_freqs(freqs_mhz)
        self.balun2.compute_for_freqs(freqs_mhz)
        self.zload.compute_for_freqs(freqs_mhz)
        #self.balun_after_vga.compute_for_freqs(freqs_mhz)

        assert self.lna.nb_freqs > 0
        assert self.lna.ABCD_matrix.shape[-1] > 0
        assert self.lna.nb_freqs==self.balun1.nb_freqs

        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)
        self.total_ABCD_matrix_nut = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

        # Note that components of ABCD_matrix for Z port of balun1 is set to 1 as no Balun is used. shape = (2,2,nports,nfreqs)
        # Note that this is a matrix multiplication
        # Associative property of matrix multiplication is used, ie. (AB)C = A(BC)
        # Make sure to multiply in this order: lna.ABCD_matrix * balun1.ABCD_matrix * cable.ABCD_matrix * vgaf.ABCD_matrix
        
        MM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        MM2 = matmul(MM1, self.lna.ABCD_matrix)
        MM3 = matmul(self.cable.ABCD_matrix, self.vgaf.ABCD_matrix)
        self.total_ABCD_matrix[:] = matmul(MM2, MM3)
        
        MMM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        self.total_ABCD_matrix_nut[:] = matmul(MMM1, self.lna.ABCD_matrix)
        
        # Calculation of Z_in (this is the total impedence of the RF chain excluding antenna arm. see page 50 of the document.)
        self.Z_load = self.zload.Z_load[np.newaxis, :] # shape (nfreq) --> (1,nfreq) to broadcast with components of ABCD_matrix with shape (2,2,ports,nfreq).
        self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])

        # Once Z_in is calculated, calculate the final total_ABCD_matrix including Balun2.
        #self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix) 

        # Antenna Impedance.
        filename = csv_files["AntennaImpedance"]["csv_file"] if csv_files["AntennaImpedance"]["enabled"] else None
        filename = grand_add_path_data(filename)
        Zant_dat = np.loadtxt(filename, delimiter=",", comments=['#', '!'], skiprows=1)
        freqs_in = Zant_dat[:,0]  # MHz
        self.Z_ant[0] = interpol_at_new_x(freqs_in, Zant_dat[:,1], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[0] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,2], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] = interpol_at_new_x(freqs_in, Zant_dat[:,3], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,4], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] = interpol_at_new_x(freqs_in, Zant_dat[:,5], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,6], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.

    def vout_f(self, voc_f):
        """ Compute final voltage after propagating signal through RF chain.
        Input: Voc_f (in frequency domain)
        Output: Voltage after RF chain in frequency domain.
        Make sure to run self.compute_for_freqs() before calling this method.
        RK Note: name 'vout_f' is a placeholder. Change it with something better. 
        """
        assert voc_f.shape==self.Z_in.shape  # shape = (nports, nfreqs)

        self.I_in_balunA = voc_f / (self.Z_ant + self.Z_in)
        self.V_in_balunA = self.I_in_balunA * self.Z_in

        # loop over three ports. shape of total_ABCD_matrix is (2,2,nports,nfreqs)
        for i in range(3):
            #ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_1port = self.total_ABCD_matrix_nut[:,:,i,:]
            ABCD_matrix_1port = np.moveaxis(ABCD_matrix_1port, -1, 0) # (2,2,nfreqs) --> (nfreqs,2,2), to compute inverse of ABCD_matrix using np.linalg.inv.
            ABCD_matrix_1port_inv = np.linalg.inv(ABCD_matrix_1port)
            V_out_RFchain = ABCD_matrix_1port_inv[:,0,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,0,1]*self.I_in_balunA[i]
            I_out_RFchain = ABCD_matrix_1port_inv[:,1,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,1,1]*self.I_in_balunA[i]

            self.V_out_RFchain[i] = V_out_RFchain
            self.I_out_RFchain[i] = I_out_RFchain

        return self.V_out_RFchain

    def get_tf(self):
        """Return transfer function for all elements in RF chain
        total transfer function is the output voltage for input Voc of 1. It says by what factor the Voc will be multiplied by the RF chain.
        @return total TF (complex, (3,N)):
        """
        self._total_tf = self.vout_f(np.ones((3, self.nb_freqs)))

        return self._total_tf
##################################################################################    

class RFChain_gaa(GenericProcessingDU):
    """
    Facade for all elements in RF chain
    """

    def __init__(self, vga_gain=0):
        r"""Assembles the RF chain of the GAA variant of the detection unit.

        Parameters
        ----------
        vga_gain : int, optional
            Gain of the variable-gain amplifier, in dB.  S-parameters are shipped
            for 20 (the GRANDProto300 default), 5, 0 and -5.

        Notes
        -----
        Construction only gathers the stages; :meth:`compute_for_freqs` evaluates
        them, and :meth:`get_tf` returns the resulting transfer function.
        """
        super().__init__()
        self.gaa = gaa_frontend0db()
        self.zload = Zload()
        # Note: self.nb_freqs at this point is 0.
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.gaa.ABCD_matrix.shape, dtype=np.complex64)

    def compute_for_freqs(self, freqs_mhz):
        """Compute transfer function for frequency freqs_mhz

        :param freqs_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        self.gaa.compute_for_freqs(freqs_mhz)
        self.zload.compute_for_freqs(freqs_mhz)
        #self.balun_after_vga.compute_for_freqs(freqs_mhz)

        assert self.gaa.nb_freqs > 0
        assert self.gaa.ABCD_matrix.shape[-1] > 0
        assert self.gaa.nb_freqs==self.gaa.nb_freqs

        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.gaa.ABCD_matrix.shape, dtype=np.complex64)
        
        self.total_ABCD_matrix[:] = self.gaa.ABCD_matrix
        
        # Calculation of Z_in (this is the total impedence of the RF chain excluding antenna arm. see page 50 of the document.)
        self.Z_load = self.zload.Z_load[np.newaxis, :] # shape (nfreq) --> (1,nfreq) to broadcast with components of ABCD_matrix with shape (2,2,ports,nfreq).
        self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])

        # Once Z_in is calculated, calculate the final total_ABCD_matrix including Balun2.
        #self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix) 

        # Antenna Impedance.
        filename = csv_files["AntennaImpedance"]["csv_file"] if csv_files["AntennaImpedance"]["enabled"] else None
        filename = grand_add_path_data(filename)
        Zant_dat = np.loadtxt(filename, delimiter=",", comments=['#', '!'], skiprows=1)
        freqs_in = Zant_dat[:,0]  # MHz
        self.Z_ant[0] = interpol_at_new_x(freqs_in, Zant_dat[:,1], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[0] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,2], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] = interpol_at_new_x(freqs_in, Zant_dat[:,3], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,4], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] = interpol_at_new_x(freqs_in, Zant_dat[:,5], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,6], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.

    def vout_f(self, voc_f):
        """ Compute final voltage after propagating signal through RF chain.
        Input: Voc_f (in frequency domain)
        Output: Voltage after RF chain in frequency domain.
        Make sure to run self.compute_for_freqs() before calling this method.
        RK Note: name 'vout_f' is a placeholder. Change it with something better. 
        """
        assert voc_f.shape==self.Z_in.shape  # shape = (nports, nfreqs)

        self.I_in_balunA = voc_f / (self.Z_ant + self.Z_in)
        self.V_in_balunA = self.I_in_balunA * self.Z_in

        # loop over three ports. shape of total_ABCD_matrix is (2,2,nports,nfreqs)
        for i in range(3):
            ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_2port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_1port = np.moveaxis(ABCD_matrix_1port, -1, 0) # (2,2,nfreqs) --> (nfreqs,2,2), to compute inverse of ABCD_matrix using np.linalg.inv.
            ABCD_matrix_1port_inv = np.linalg.inv(ABCD_matrix_1port)
            
            self.V_out_RFchain[i] = 2/(self.total_ABCD_matrix[0,0,i,:] + self.total_ABCD_matrix[0,1,i,:]/50 + self.total_ABCD_matrix[1,0,i,:]*50 + self.total_ABCD_matrix[1,1,i,:])
            #V_out_RFchain = ABCD_matrix_1port_inv[:,0,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,0,1]*self.I_in_balunA[i]
            I_out_RFchain = ABCD_matrix_1port_inv[:,1,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,1,1]*self.I_in_balunA[i]
            #V_out_RFchain = ABCD_matrix_1port_inv[:,0,0] + ABCD_matrix_1port_inv[:,0,1]
            #I_out_RFchain = ABCD_matrix_1port_inv[:,1,0] + ABCD_matrix_1port_inv[:,1,1]

            #self.V_out_RFchain[i] = V_out_RFchain
            self.I_out_RFchain[i] = I_out_RFchain

        return self.V_out_RFchain

    def get_tf(self):
        """Return transfer function for all elements in RF chain
        total transfer function is the output voltage for input Voc of 1. It says by what factor the Voc will be multiplied by the RF chain.
        @return total TF (complex, (3,N)):
        """
        self._total_tf = self.vout_f(np.ones((3, self.nb_freqs)))

        return self._total_tf
##########################################################################################
###########################################################################################
    
class RFChain_Balun1(GenericProcessingDU):
    """
    Facade for all elements in RF chain
    """

    def __init__(self, vga_gain=20):
        r"""Assembles the chain truncated after the first balun.

        Parameters
        ----------
        vga_gain : int, optional
            Gain of the variable-gain amplifier, in dB.  S-parameters are shipped
            for 20 (the GRANDProto300 default), 5, 0 and -5.

        Useful for isolating one stage's contribution.

        Notes
        -----
        Construction only gathers the stages; :meth:`compute_for_freqs` evaluates
        them, and :meth:`get_tf` returns the resulting transfer function.
        """
        super().__init__()
        self.matcnet = MatchingNetwork()
        self.lna = LowNoiseAmplifier()
        self.balun1 = BalunAfterLNA()
        self.cable = Cable()
        self.vgaf = VGAFilter(gain=vga_gain)
        self.balun2 = BalunBeforeADC()
        self.zload = Zload()
        # Note: self.nb_freqs at this point is 0.
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

    def compute_for_freqs(self, freqs_mhz):
        """Compute transfer function for frequency freqs_mhz

        :param freqs_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        self.matcnet.compute_for_freqs(freqs_mhz)
        self.lna.compute_for_freqs(freqs_mhz)
        self.balun1.compute_for_freqs(freqs_mhz)
        self.cable.compute_for_freqs(freqs_mhz)
        self.vgaf.compute_for_freqs(freqs_mhz)
        self.balun2.compute_for_freqs(freqs_mhz)
        self.zload.compute_for_freqs(freqs_mhz)
        #self.balun_after_vga.compute_for_freqs(freqs_mhz)

        assert self.lna.nb_freqs > 0
        assert self.lna.ABCD_matrix.shape[-1] > 0
        assert self.lna.nb_freqs==self.balun1.nb_freqs

        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)
        self.total_ABCD_matrix_nut = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

        # Note that components of ABCD_matrix for Z port of balun1 is set to 1 as no Balun is used. shape = (2,2,nports,nfreqs)
        # Note that this is a matrix multiplication
        # Associative property of matrix multiplication is used, ie. (AB)C = A(BC)
        # Make sure to multiply in this order: lna.ABCD_matrix * balun1.ABCD_matrix * cable.ABCD_matrix * vgaf.ABCD_matrix
        
        MM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        MM2 = matmul(MM1, self.lna.ABCD_matrix)
        MM3 = matmul(self.cable.ABCD_matrix, self.vgaf.ABCD_matrix)
        self.total_ABCD_matrix[:] = matmul(MM2, MM3)
        
        self.total_ABCD_matrix_nut[:] = self.balun1.ABCD_matrix
        
        # Calculation of Z_in (this is the total impedence of the RF chain excluding antenna arm. see page 50 of the document.)
        self.Z_load = self.zload.Z_load[np.newaxis, :] # shape (nfreq) --> (1,nfreq) to broadcast with components of ABCD_matrix with shape (2,2,ports,nfreq).
        self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])

        # Once Z_in is calculated, calculate the final total_ABCD_matrix including Balun2.
        #self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix) 

        # Antenna Impedance.
        filename = csv_files["AntennaImpedance"]["csv_file"] if csv_files["AntennaImpedance"]["enabled"] else None
        filename = grand_add_path_data(filename)
        Zant_dat = np.loadtxt(filename, delimiter=",", comments=['#', '!'], skiprows=1)
        freqs_in = Zant_dat[:,0]  # MHz
        self.Z_ant[0] = interpol_at_new_x(freqs_in, Zant_dat[:,1], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[0] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,2], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] = interpol_at_new_x(freqs_in, Zant_dat[:,3], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,4], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] = interpol_at_new_x(freqs_in, Zant_dat[:,5], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,6], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.

    def vout_f(self, voc_f):
        """ Compute final voltage after propagating signal through RF chain.
        Input: Voc_f (in frequency domain)
        Output: Voltage after RF chain in frequency domain.
        Make sure to run self.compute_for_freqs() before calling this method.
        RK Note: name 'vout_f' is a placeholder. Change it with something better. 
        """
        assert voc_f.shape==self.Z_in.shape  # shape = (nports, nfreqs)

        self.I_in_balunA = voc_f / (self.Z_ant + self.Z_in)
        self.V_in_balunA = self.I_in_balunA * self.Z_in

        # loop over three ports. shape of total_ABCD_matrix is (2,2,nports,nfreqs)
        for i in range(3):
            #ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_1port = self.total_ABCD_matrix_nut[:,:,i,:]
            ABCD_matrix_1port = np.moveaxis(ABCD_matrix_1port, -1, 0) # (2,2,nfreqs) --> (nfreqs,2,2), to compute inverse of ABCD_matrix using np.linalg.inv.
            ABCD_matrix_1port_inv = np.linalg.inv(ABCD_matrix_1port)
            V_out_RFchain = ABCD_matrix_1port_inv[:,0,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,0,1]*self.I_in_balunA[i]
            I_out_RFchain = ABCD_matrix_1port_inv[:,1,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,1,1]*self.I_in_balunA[i]

            self.V_out_RFchain[i] = V_out_RFchain
            self.I_out_RFchain[i] = I_out_RFchain

        return self.V_out_RFchain

    def get_tf(self):
        """Return transfer function for all elements in RF chain
        total transfer function is the output voltage for input Voc of 1. It says by what factor the Voc will be multiplied by the RF chain.
        @return total TF (complex, (3,N)):
        """
        self._total_tf = self.vout_f(np.ones((3, self.nb_freqs)))

        return self._total_tf
################################################################################## 
###########################################################################################
    
class RFChain_Match_net(GenericProcessingDU):
    """
    Facade for all elements in RF chain
    """

    def __init__(self, vga_gain=20):
        r"""Assembles the chain truncated after the matching network.

        Parameters
        ----------
        vga_gain : int, optional
            Gain of the variable-gain amplifier, in dB.  S-parameters are shipped
            for 20 (the GRANDProto300 default), 5, 0 and -5.

        Useful for isolating one stage's contribution.

        Notes
        -----
        Construction only gathers the stages; :meth:`compute_for_freqs` evaluates
        them, and :meth:`get_tf` returns the resulting transfer function.
        """
        super().__init__()
        self.matcnet = MatchingNetwork()
        self.lna = LowNoiseAmplifier()
        self.balun1 = BalunAfterLNA()
        self.cable = Cable()
        self.vgaf = VGAFilter(gain=vga_gain)
        self.balun2 = BalunBeforeADC()
        self.zload = Zload()
        # Note: self.nb_freqs at this point is 0.
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

    def compute_for_freqs(self, freqs_mhz):
        """Compute transfer function for frequency freqs_mhz

        :param freqs_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        self.matcnet.compute_for_freqs(freqs_mhz)
        self.lna.compute_for_freqs(freqs_mhz)
        self.balun1.compute_for_freqs(freqs_mhz)
        self.cable.compute_for_freqs(freqs_mhz)
        self.vgaf.compute_for_freqs(freqs_mhz)
        self.balun2.compute_for_freqs(freqs_mhz)
        self.zload.compute_for_freqs(freqs_mhz)
        #self.balun_after_vga.compute_for_freqs(freqs_mhz)

        assert self.lna.nb_freqs > 0
        assert self.lna.ABCD_matrix.shape[-1] > 0
        assert self.lna.nb_freqs==self.balun1.nb_freqs

        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)
        self.total_ABCD_matrix_nut = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

        # Note that components of ABCD_matrix for Z port of balun1 is set to 1 as no Balun is used. shape = (2,2,nports,nfreqs)
        # Note that this is a matrix multiplication
        # Associative property of matrix multiplication is used, ie. (AB)C = A(BC)
        # Make sure to multiply in this order: lna.ABCD_matrix * balun1.ABCD_matrix * cable.ABCD_matrix * vgaf.ABCD_matrix
        
        MM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        MM2 = matmul(MM1, self.lna.ABCD_matrix)
        MM3 = matmul(self.cable.ABCD_matrix, self.vgaf.ABCD_matrix)
        self.total_ABCD_matrix[:] = matmul(MM2, MM3)
        
        self.total_ABCD_matrix_nut[:] = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        
        # Calculation of Z_in (this is the total impedence of the RF chain excluding antenna arm. see page 50 of the document.)
        self.Z_load = self.zload.Z_load[np.newaxis, :] # shape (nfreq) --> (1,nfreq) to broadcast with components of ABCD_matrix with shape (2,2,ports,nfreq).
        self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])

        # Once Z_in is calculated, calculate the final total_ABCD_matrix including Balun2.
        #self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix) 

        # Antenna Impedance.
        filename = csv_files["AntennaImpedance"]["csv_file"] if csv_files["AntennaImpedance"]["enabled"] else None
        filename = grand_add_path_data(filename)
        Zant_dat = np.loadtxt(filename, delimiter=",", comments=['#', '!'], skiprows=1)
        freqs_in = Zant_dat[:,0]  # MHz
        self.Z_ant[0] = interpol_at_new_x(freqs_in, Zant_dat[:,1], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[0] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,2], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] = interpol_at_new_x(freqs_in, Zant_dat[:,3], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,4], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] = interpol_at_new_x(freqs_in, Zant_dat[:,5], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,6], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.

    def vout_f(self, voc_f):
        """ Compute final voltage after propagating signal through RF chain.
        Input: Voc_f (in frequency domain)
        Output: Voltage after RF chain in frequency domain.
        Make sure to run self.compute_for_freqs() before calling this method.
        RK Note: name 'vout_f' is a placeholder. Change it with something better. 
        """
        assert voc_f.shape==self.Z_in.shape  # shape = (nports, nfreqs)

        self.I_in_balunA = voc_f / (self.Z_ant + self.Z_in)
        self.V_in_balunA = self.I_in_balunA * self.Z_in

        # loop over three ports. shape of total_ABCD_matrix is (2,2,nports,nfreqs)
        for i in range(3):
            #ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_1port = self.total_ABCD_matrix_nut[:,:,i,:]
            ABCD_matrix_1port = np.moveaxis(ABCD_matrix_1port, -1, 0) # (2,2,nfreqs) --> (nfreqs,2,2), to compute inverse of ABCD_matrix using np.linalg.inv.
            ABCD_matrix_1port_inv = np.linalg.inv(ABCD_matrix_1port)
            V_out_RFchain = ABCD_matrix_1port_inv[:,0,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,0,1]*self.I_in_balunA[i]
            I_out_RFchain = ABCD_matrix_1port_inv[:,1,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,1,1]*self.I_in_balunA[i]

            self.V_out_RFchain[i] = V_out_RFchain
            self.I_out_RFchain[i] = I_out_RFchain

        return self.V_out_RFchain

    def get_tf(self):
        """Return transfer function for all elements in RF chain
        total transfer function is the output voltage for input Voc of 1. It says by what factor the Voc will be multiplied by the RF chain.
        @return total TF (complex, (3,N)):
        """
        self._total_tf = self.vout_f(np.ones((3, self.nb_freqs)))

        return self._total_tf
##################################################################################    

class RFChain_Cable_Connectors(GenericProcessingDU):
    """
    Facade for all elements in RF chain
    """

    def __init__(self, vga_gain=20):
        r"""Assembles the chain truncated after the cable and connectors.

        Parameters
        ----------
        vga_gain : int, optional
            Gain of the variable-gain amplifier, in dB.  S-parameters are shipped
            for 20 (the GRANDProto300 default), 5, 0 and -5.

        Useful for isolating one stage's contribution.

        Notes
        -----
        Construction only gathers the stages; :meth:`compute_for_freqs` evaluates
        them, and :meth:`get_tf` returns the resulting transfer function.
        """
        super().__init__()
        self.matcnet = MatchingNetwork()
        self.lna = LowNoiseAmplifier()
        self.balun1 = BalunAfterLNA()
        self.cable = Cable()
        self.vgaf = VGAFilter(gain=vga_gain)
        self.balun2 = BalunBeforeADC()
        self.zload = Zload()
        # Note: self.nb_freqs at this point is 0.
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

    def compute_for_freqs(self, freqs_mhz):
        """Compute transfer function for frequency freqs_mhz

        :param freqs_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        self.matcnet.compute_for_freqs(freqs_mhz)
        self.lna.compute_for_freqs(freqs_mhz)
        self.balun1.compute_for_freqs(freqs_mhz)
        self.cable.compute_for_freqs(freqs_mhz)
        self.vgaf.compute_for_freqs(freqs_mhz)
        self.balun2.compute_for_freqs(freqs_mhz)
        self.zload.compute_for_freqs(freqs_mhz)
        #self.balun_after_vga.compute_for_freqs(freqs_mhz)

        assert self.lna.nb_freqs > 0
        assert self.lna.ABCD_matrix.shape[-1] > 0
        assert self.lna.nb_freqs==self.balun1.nb_freqs

        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)
        self.total_ABCD_matrix_nut = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

        # Note that components of ABCD_matrix for Z port of balun1 is set to 1 as no Balun is used. shape = (2,2,nports,nfreqs)
        # Note that this is a matrix multiplication
        # Associative property of matrix multiplication is used, ie. (AB)C = A(BC)
        # Make sure to multiply in this order: lna.ABCD_matrix * balun1.ABCD_matrix * cable.ABCD_matrix * vgaf.ABCD_matrix
        
        MM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        MM2 = matmul(MM1, self.lna.ABCD_matrix)
        MM3 = matmul(self.cable.ABCD_matrix, self.vgaf.ABCD_matrix)
        self.total_ABCD_matrix[:] = matmul(MM2, MM3)
        
        MMM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        MMM2 = matmul(MMM1, self.lna.ABCD_matrix)
        self.total_ABCD_matrix_nut[:] = matmul(MMM2, self.cable.ABCD_matrix)
        
        # Calculation of Z_in (this is the total impedence of the RF chain excluding antenna arm. see page 50 of the document.)
        self.Z_load = self.zload.Z_load[np.newaxis, :] # shape (nfreq) --> (1,nfreq) to broadcast with components of ABCD_matrix with shape (2,2,ports,nfreq).
        self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])

        # Once Z_in is calculated, calculate the final total_ABCD_matrix including Balun2.
        #self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix) 

        # Antenna Impedance.
        filename = csv_files["AntennaImpedance"]["csv_file"] if csv_files["AntennaImpedance"]["enabled"] else None
        filename = grand_add_path_data(filename)
        Zant_dat = np.loadtxt(filename, delimiter=",", comments=['#', '!'], skiprows=1)
        freqs_in = Zant_dat[:,0]  # MHz
        self.Z_ant[0] = interpol_at_new_x(freqs_in, Zant_dat[:,1], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[0] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,2], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] = interpol_at_new_x(freqs_in, Zant_dat[:,3], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,4], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] = interpol_at_new_x(freqs_in, Zant_dat[:,5], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,6], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.

    def vout_f(self, voc_f):
        """ Compute final voltage after propagating signal through RF chain.
        Input: Voc_f (in frequency domain)
        Output: Voltage after RF chain in frequency domain.
        Make sure to run self.compute_for_freqs() before calling this method.
        RK Note: name 'vout_f' is a placeholder. Change it with something better. 
        """
        assert voc_f.shape==self.Z_in.shape  # shape = (nports, nfreqs)

        self.I_in_balunA = voc_f / (self.Z_ant + self.Z_in)
        self.V_in_balunA = self.I_in_balunA * self.Z_in

        # loop over three ports. shape of total_ABCD_matrix is (2,2,nports,nfreqs)
        for i in range(3):
            #ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_1port = self.total_ABCD_matrix_nut[:,:,i,:]
            ABCD_matrix_1port = np.moveaxis(ABCD_matrix_1port, -1, 0) # (2,2,nfreqs) --> (nfreqs,2,2), to compute inverse of ABCD_matrix using np.linalg.inv.
            ABCD_matrix_1port_inv = np.linalg.inv(ABCD_matrix_1port)
            V_out_RFchain = ABCD_matrix_1port_inv[:,0,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,0,1]*self.I_in_balunA[i]
            I_out_RFchain = ABCD_matrix_1port_inv[:,1,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,1,1]*self.I_in_balunA[i]

            self.V_out_RFchain[i] = V_out_RFchain
            self.I_out_RFchain[i] = I_out_RFchain

        return self.V_out_RFchain

    def get_tf(self):
        """Return transfer function for all elements in RF chain
        total transfer function is the output voltage for input Voc of 1. It says by what factor the Voc will be multiplied by the RF chain.
        @return total TF (complex, (3,N)):
        """
        self._total_tf = self.vout_f(np.ones((3, self.nb_freqs)))

        return self._total_tf
##################################################################################  

class RFChain_VGA(GenericProcessingDU):
    """
    Facade for all elements in RF chain
    """

    def __init__(self, vga_gain=20):
        r"""Assembles the chain truncated after the variable-gain amplifier.

        Parameters
        ----------
        vga_gain : int, optional
            Gain of the variable-gain amplifier, in dB.  S-parameters are shipped
            for 20 (the GRANDProto300 default), 5, 0 and -5.

        Useful for isolating one stage's contribution.

        Notes
        -----
        Construction only gathers the stages; :meth:`compute_for_freqs` evaluates
        them, and :meth:`get_tf` returns the resulting transfer function.
        """
        super().__init__()
        self.matcnet = MatchingNetwork()
        self.lna = LowNoiseAmplifier()
        self.balun1 = BalunAfterLNA()
        self.cable = Cable()
        self.vgaf = VGAFilter(gain=vga_gain)
        self.balun2 = BalunBeforeADC()
        self.zload = Zload()
        # Note: self.nb_freqs at this point is 0.
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

    def compute_for_freqs(self, freqs_mhz):
        """Compute transfer function for frequency freqs_mhz

        :param freqs_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        self.matcnet.compute_for_freqs(freqs_mhz)
        self.lna.compute_for_freqs(freqs_mhz)
        self.balun1.compute_for_freqs(freqs_mhz)
        self.cable.compute_for_freqs(freqs_mhz)
        self.vgaf.compute_for_freqs(freqs_mhz)
        self.balun2.compute_for_freqs(freqs_mhz)
        self.zload.compute_for_freqs(freqs_mhz)
        #self.balun_after_vga.compute_for_freqs(freqs_mhz)

        assert self.lna.nb_freqs > 0
        assert self.lna.ABCD_matrix.shape[-1] > 0
        assert self.lna.nb_freqs==self.balun1.nb_freqs

        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)
        self.total_ABCD_matrix_nut = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

        # Note that components of ABCD_matrix for Z port of balun1 is set to 1 as no Balun is used. shape = (2,2,nports,nfreqs)
        # Note that this is a matrix multiplication
        # Associative property of matrix multiplication is used, ie. (AB)C = A(BC)
        # Make sure to multiply in this order: lna.ABCD_matrix * balun1.ABCD_matrix * cable.ABCD_matrix * vgaf.ABCD_matrix
        
        MM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        MM2 = matmul(MM1, self.lna.ABCD_matrix)
        MM3 = matmul(self.cable.ABCD_matrix, self.vgaf.ABCD_matrix)
        self.total_ABCD_matrix[:] = matmul(MM2, MM3)
        
        # Calculation of Z_in (this is the total impedence of the RF chain excluding antenna arm. see page 50 of the document.)
        self.Z_load = self.zload.Z_load[np.newaxis, :] # shape (nfreq) --> (1,nfreq) to broadcast with components of ABCD_matrix with shape (2,2,ports,nfreq).
        self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])

        # Once Z_in is calculated, calculate the final total_ABCD_matrix including Balun2.
        #self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix) 

        # Antenna Impedance.
        filename = csv_files["AntennaImpedance"]["csv_file"] if csv_files["AntennaImpedance"]["enabled"] else None
        filename = grand_add_path_data(filename)
        Zant_dat = np.loadtxt(filename, delimiter=",", comments=['#', '!'], skiprows=1)
        freqs_in = Zant_dat[:,0]  # MHz
        self.Z_ant[0] = interpol_at_new_x(freqs_in, Zant_dat[:,1], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[0] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,2], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] = interpol_at_new_x(freqs_in, Zant_dat[:,3], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,4], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] = interpol_at_new_x(freqs_in, Zant_dat[:,5], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,6], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.

    def vout_f(self, voc_f):
        """ Compute final voltage after propagating signal through RF chain.
        Input: Voc_f (in frequency domain)
        Output: Voltage after RF chain in frequency domain.
        Make sure to run self.compute_for_freqs() before calling this method.
        RK Note: name 'vout_f' is a placeholder. Change it with something better. 
        """
        assert voc_f.shape==self.Z_in.shape  # shape = (nports, nfreqs)

        self.I_in_balunA = voc_f / (self.Z_ant + self.Z_in)
        self.V_in_balunA = self.I_in_balunA * self.Z_in

        # loop over three ports. shape of total_ABCD_matrix is (2,2,nports,nfreqs)
        for i in range(3):
            #ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_1port = np.moveaxis(ABCD_matrix_1port, -1, 0) # (2,2,nfreqs) --> (nfreqs,2,2), to compute inverse of ABCD_matrix using np.linalg.inv.
            ABCD_matrix_1port_inv = np.linalg.inv(ABCD_matrix_1port)
            V_out_RFchain = ABCD_matrix_1port_inv[:,0,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,0,1]*self.I_in_balunA[i]
            I_out_RFchain = ABCD_matrix_1port_inv[:,1,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,1,1]*self.I_in_balunA[i]

            self.V_out_RFchain[i] = V_out_RFchain
            self.I_out_RFchain[i] = I_out_RFchain

        return self.V_out_RFchain

    def get_tf(self):
        """Return transfer function for all elements in RF chain
        total transfer function is the output voltage for input Voc of 1. It says by what factor the Voc will be multiplied by the RF chain.
        @return total TF (complex, (3,N)):
        """
        self._total_tf = self.vout_f(np.ones((3, self.nb_freqs)))

        return self._total_tf
##################################################################################    
class RFChain_in_Balun1(GenericProcessingDU):
    """
    Facade for all elements in RF chain
    """

    def __init__(self, vga_gain=20):
        r"""Assembles the chain up to the input of the first balun.

        Parameters
        ----------
        vga_gain : int, optional
            Gain of the variable-gain amplifier, in dB.  S-parameters are shipped
            for 20 (the GRANDProto300 default), 5, 0 and -5.

        Useful for isolating one stage's contribution.

        Notes
        -----
        Construction only gathers the stages; :meth:`compute_for_freqs` evaluates
        them, and :meth:`get_tf` returns the resulting transfer function.
        """
        super().__init__()
        self.matcnet = MatchingNetwork()
        self.lna = LowNoiseAmplifier()
        self.balun1 = BalunAfterLNA()
        self.cable = Cable()
        self.vgaf = VGAFilter(gain=vga_gain)
        self.balun2 = BalunBeforeADC()
        self.zload = Zload()
        # Note: self.nb_freqs at this point is 0.
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

    def compute_for_freqs(self, freqs_mhz):
        """Compute transfer function for frequency freqs_mhz

        :param freqs_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        self.matcnet.compute_for_freqs(freqs_mhz)
        self.lna.compute_for_freqs(freqs_mhz)
        self.balun1.compute_for_freqs(freqs_mhz)
        self.cable.compute_for_freqs(freqs_mhz)
        self.vgaf.compute_for_freqs(freqs_mhz)
        self.balun2.compute_for_freqs(freqs_mhz)
        self.zload.compute_for_freqs(freqs_mhz)
        #self.balun_after_vga.compute_for_freqs(freqs_mhz)

        assert self.lna.nb_freqs > 0
        assert self.lna.ABCD_matrix.shape[-1] > 0
        assert self.lna.nb_freqs==self.balun1.nb_freqs

        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)
        self.total_ABCD_matrix_nut = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

        # Note that components of ABCD_matrix for Z port of balun1 is set to 1 as no Balun is used. shape = (2,2,nports,nfreqs)
        # Note that this is a matrix multiplication
        # Associative property of matrix multiplication is used, ie. (AB)C = A(BC)
        # Make sure to multiply in this order: lna.ABCD_matrix * balun1.ABCD_matrix * cable.ABCD_matrix * vgaf.ABCD_matrix
        
        MM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        MM2 = matmul(MM1, self.lna.ABCD_matrix)
        MM3 = matmul(self.cable.ABCD_matrix, self.vgaf.ABCD_matrix)
        self.total_ABCD_matrix[:] = matmul(MM2, MM3)
        
        self.total_ABCD_matrix_nut[:] = np.ones(self.lna.ABCD_matrix.shape, dtype=np.complex64)
        
        # Calculation of Z_in (this is the total impedence of the RF chain excluding antenna arm. see page 50 of the document.)
        self.Z_load = self.zload.Z_load[np.newaxis, :] # shape (nfreq) --> (1,nfreq) to broadcast with components of ABCD_matrix with shape (2,2,ports,nfreq).
        self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])

        # Once Z_in is calculated, calculate the final total_ABCD_matrix including Balun2.
        #self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix) 

        # Antenna Impedance.
        filename = csv_files["AntennaImpedance"]["csv_file"] if csv_files["AntennaImpedance"]["enabled"] else None
        filename = grand_add_path_data(filename)
        Zant_dat = np.loadtxt(filename, delimiter=",", comments=['#', '!'], skiprows=1)
        freqs_in = Zant_dat[:,0]  # MHz
        self.Z_ant[0] = interpol_at_new_x(freqs_in, Zant_dat[:,1], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[0] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,2], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] = interpol_at_new_x(freqs_in, Zant_dat[:,3], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,4], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] = interpol_at_new_x(freqs_in, Zant_dat[:,5], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,6], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.

    def vout_f(self, voc_f):
        """ Compute final voltage after propagating signal through RF chain.
        Input: Voc_f (in frequency domain)
        Output: Voltage after RF chain in frequency domain.
        Make sure to run self.compute_for_freqs() before calling this method.
        RK Note: name 'vout_f' is a placeholder. Change it with something better. 
        """
        assert voc_f.shape==self.Z_in.shape  # shape = (nports, nfreqs)

        self.I_in_balunA = voc_f / (self.Z_ant + self.Z_in)
        self.V_in_balunA = self.I_in_balunA * self.Z_in

        # loop over three ports. shape of total_ABCD_matrix is (2,2,nports,nfreqs)
        for i in range(3):
            #ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            #ABCD_matrix_1port = self.total_ABCD_matrix_nut[:,:,i,:]
            #ABCD_matrix_1port = np.moveaxis(ABCD_matrix_1port, -1, 0) # (2,2,nfreqs) --> (nfreqs,2,2), to compute inverse of ABCD_matrix using np.linalg.inv.
            #ABCD_matrix_1port_inv = np.linalg.inv(ABCD_matrix_1port)
            #V_out_RFchain = ABCD_matrix_1port_inv[:,0,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,0,1]*self.I_in_balunA[i]
            #I_out_RFchain = ABCD_matrix_1port_inv[:,1,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,1,1]*self.I_in_balunA[i]
            V_out_RFchain = 1*self.V_in_balunA[i] + 1*self.I_in_balunA[i]
            I_out_RFchain = 1*self.V_in_balunA[i] + 1*self.I_in_balunA[i]

            self.V_out_RFchain[i] = V_out_RFchain
            self.I_out_RFchain[i] = I_out_RFchain

        return self.V_out_RFchain

    def get_tf(self):
        """Return transfer function for all elements in RF chain
        total transfer function is the output voltage for input Voc of 1. It says by what factor the Voc will be multiplied by the RF chain.
        @return total TF (complex, (3,N)):
        """
        self._total_tf = self.vout_f(np.ones((3, self.nb_freqs)))

        return self._total_tf
################################################################################## 