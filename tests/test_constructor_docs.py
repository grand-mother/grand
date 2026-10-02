"""Every dataclass constructor documents its parameters (#261, item 10).

The fields are documented where they are declared; ``grand.basis.fielddoc``
lists them in each class's Parameters section, so ``help()`` and the API
reference show them.
"""
import dataclasses

import pytest

from grand.basis.fielddoc import document_fields
from grand.dataio import event_trees, run_trees
from grand.dataio.data_tree import DataTree

#: DAQ firmware parameters whose meaning the source does not record; they are
#: listed with their type, and wait on the firmware's documentation.
FIRMWARE = {"TADC", "TRunRawVoltage"}


def _classes():
    import grand.aoi.antenna as an
    import grand.aoi.event as ev
    import grand.aoi.shower as sh
    import grand.aoi.timetrace as tt
    import grand.basis.type_trace as ty
    import grand.sim.detector.antenna_model as am
    import grand.sim.detector.process_ant as pa
    import grand.sim.shower.gen_shower as gs

    trees = [c for m in (event_trees, run_trees) for c in vars(m).values()
             if isinstance(c, type) and issubclass(c, DataTree) and c.__module__ == m.__name__]
    return trees + [ev.Event, sh.Shower, an.Antenna, tt.Timetrace3D, tt.Voltage, tt.Efield,
                    ty.ElectricField, ty.Voltage, am.DataTable, pa.PreComputeInterpol,
                    pa.AntennaProcessing, gs.ShowerEvent]


@pytest.mark.parametrize("cls", _classes(), ids=lambda c: c.__name__)
def test_every_field_is_a_documented_parameter(cls):
    doc = cls.__doc__
    assert "\nParameters\n----------\n" in doc
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue
        start = doc.find("\n%s : " % f.name)
        assert start >= 0, "%s.%s is not listed" % (cls.__name__, f.name)
        if cls.__name__ not in FIRMWARE:
            following = doc[start + 1:].split("\n")
            assert len(following) > 1 and following[1].startswith("    "), \
                "%s.%s has no description" % (cls.__name__, f.name)


def test_types_and_units_come_from_the_tree_descriptors():
    doc = event_trees.TShower.__doc__
    assert "\nenergy_primary : float32, in GeV\n" in doc
    assert "\nprimary_type : str\n" in doc


def test_a_documented_class_is_left_alone():
    @document_fields
    @dataclasses.dataclass
    class Done:
        """Done.

        Parameters
        ----------
        x : int
            Said already.
        """
        x: int = 0
        """Not repeated"""

    assert Done.__doc__.count("Parameters") == 1 and "Not repeated" not in Done.__doc__


def test_comments_and_strings_both_count_and_code_is_skipped():
    @document_fields
    @dataclasses.dataclass
    class Fields:
        """Fields."""
        ## From a double-hash comment
        a: int = 0
        # old: int = 1
        # From a single-hash comment
        b: int = 0
        c: int = 0
        """From a string"""

    doc = Fields.__doc__
    assert "a : int\n    From a double-hash comment" in doc
    assert "b : int\n    From a single-hash comment" in doc and "old" not in doc
    assert "c : int\n    From a string" in doc


def test_the_parameters_come_before_the_examples():
    r"""numpydoc renders Examples last; a tree class with an example keeps that order."""
    from grand.dataio.event_trees import TShower

    doc = TShower.__doc__
    assert doc.index("\nParameters\n") < doc.index("\nSee Also\n") < doc.index("\nExamples\n")
    assert doc.count("\nExamples\n") == 1 and doc.count("\nSee Also\n") == 1
