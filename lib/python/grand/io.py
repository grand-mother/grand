from __future__ import annotations

from collections import OrderedDict
import os
from typing import Optional, Tuple, Union

import astropy.units as u
from astropy.coordinates import CartesianRepresentation
import h5py
from h5py import Dataset as _Dataset, File as _File, Group as _Group
import numpy

__all__ = ["DataNode", "ElementsIterator"]


class ElementsIterator:
    """
    Iterator over the data elements of a node
    """
    def __init__(self, node: DataNode) -> None:
        self.node = node

    def __iter__(self) -> ElementsIterator:
        self._data = self.node._group.__iter__()
        return self

    def __next__(self) -> Tuple[str, DataNode]:
        while True:
            k: str = self._data.__next__()
            v: DataNode = self.node._group[k]
            if type(v) == _Dataset:
                return k, self.node._unpack(v)


class DataNode:
    """
    A node containing data elements and branches to sub-nodes
    """

    _compression = {"compression": "gzip", "compression_opts": 9}

    def __init__(self, group: _Group) -> None:
        self._group = group
        self._name: Optional[str] = None

    def __getitem__(self, k: str) -> DataNode:
        v = self._group[k]
        if type(v) == _Group:
            return DataNode(v)
        else:
            raise KeyError(k)

    def __iter__(self) -> DataNode:
        self._group_iter = self._group.__iter__()
        return self

    def __next__(self) -> DataNode:
        while True:
            k = self._group_iter.__next__()
            v = self._group[k]
            if type(v) == _Group:
                return DataNode(v)

    def __enter__(self) -> DataNode:
        return self

    def __exit__(self, *args) -> None:
        pass

    def branch(self, k: str) -> DataNode:
        v:_Group = self._group.require_group(k)
        return DataNode(v)

    def close(self) -> None:
        self._group.file.close()

    def read(self, *args: str, dtype: Union[numpy.DataType, str, None]=None):
        res = len(args) * [None]
        for i, k in enumerate(args):
            v = self._group[k]
            if type(v) == _Dataset:
                res[i] = self._unpack(v, dtype)
            else:
                raise KeyError(k)
        if len(res) == 1:
            return res[0]
        else:
            return res

    def write(self, k, v, dtype=None, unit=None, columns=None, units=None):
        if isinstance(v, (str, bytes, numpy.string_, numpy.bytes_)):
            dset = self._write_string(k, v)
        elif isinstance(v, u.Quantity):
            dset = self._write_quantity(k, v, dtype, unit)
        elif isinstance(v, CartesianRepresentation):
            dset = self._write_cartesian(k, v, dtype, columns, unit)
        elif isinstance(v, (list, tuple)):
            dset = self._write_table(k, v, dtype, columns, units)
        elif isinstance(v, numpy.ndarray):
            if columns:
                self._check_columns(v, columns)
            if units:
                self._check_units(v, units)
            dset = self._write_array(k, v, dtype, columns, units)
        else:
            dset = self._write_number(k, v, dtype, unit)

    def _unpack(self, dset: _Dataset,
                      dtype: Union[numpy.DataType, str, None]=None):
        if dset.shape:
            v = dset[:]
            if dtype is not None:
                v = v.astype(dtype)
        else:
            v = dset[()]
            if dtype is not None:
                v = numpy.dtype(dtype).type(v)

        if type(v) == numpy.bytes_:
            return self._unpack_string(dset, v)

        try:
            metatype = dset.attrs["metatype"]
        except KeyError:
            metatype = None

        if metatype is None:
            return v
        elif metatype == "quantity":
            return self._unpack_quantity(dset, v)
        elif metatype == "cartesian":
            return self._unpack_cartesian(dset, v)
        elif metatype == "table":
            return self._unpack_table(dset, v)
        else:
            raise ValueError(f"Invalid metatype {metatype}")

        return v

    def _write_string(self, k, v, columns=None, units=None):
        if hasattr(v, "encode"):
            encoding = "UTF-8"
            v = v.encode()
        else:
            encoding = "ASCII"

        v = numpy.bytes_(v)

        if len(v) > 128:
            opts = self._compression
        else:
            opts = {}
        dset = self._group.require_dataset(k, data=v, shape=v.shape,
                                           dtype=v.dtype, **opts)
        dset.attrs["encoding"] = encoding

        return dset

    @staticmethod
    def _unpack_string(dset, v):
        encoding = dset.attrs["encoding"]
        if encoding != "ASCII":
            return v.decode(encoding)
        else:
            return v.tobytes()

    def _write_quantity(self, k, v, dtype=None, unit=None):
        if dtype is None:
            dtype = v.dtype

        if unit is None:
            unit = str(v.unit)
        v = v.to_value(unit)

        try:
            shape = v.shape
        except AttributeError:
            shape = ()

        dset = self._group.require_dataset(k, data=v, dtype=dtype,
                                           shape=shape)
        dset.attrs["unit"] = unit
        dset.attrs["metatype"] = "quantity"

        return dset

    @staticmethod
    def _unpack_quantity(dset, v):
        unit = dset.attrs["unit"]
        return v * u.Unit(unit)

    def _write_cartesian(self, k, v, dtype=None, columns=None, unit=None):
        v = v.xyz

        if columns is None:
            columns = ("x", "y", "z")
        else:
            self._check_columns(v, columns)

        if unit is None:
            unit = str(v.unit)
        units = (unit, unit, unit)

        dset = self._write_array(k, v.value, dtype, columns, units)
        dset.attrs["metatype"] = "cartesian"

        return dset

    @staticmethod
    def _unpack_cartesian(dset, v):
        unit = dset.attrs["units"][0]
        v *= u.Unit(unit)
        return CartesianRepresentation(v)

    def _write_table(self, k, v, dtype=None, columns=None, units=None):
        if columns:
            self._check_columns(v, columns)

        if units:
            self._check_units(v, units)
        else:
            units = [self._get_unit(vi) for vi in v]

        if dtype is None:
            dtype = v[0].dtype

        n = len(v)
        try:
            m = len(v[0])
        except TypeError:
            m = 1

        data = numpy.zeros((n, m), dtype=dtype)

        for i, ui in enumerate(units):
            if ui:
                data[i,:] = v[i].to_value(ui)
            else:
                data[i,:] = v[i]

        dset = self._write_array(k, data, dtype, columns, units)
        dset.attrs["metatype"] = "table"

        return dset

    @staticmethod
    def _unpack_table(dset, v):
        units = dset.attrs["units"]
        table = []
        for i, ui in enumerate(units):
            vi = v[i,:]
            if vi.size == 1:
                vi = vi[0]
            if ui:
                vi *= u.Unit(ui)
            table.append(vi)
        return table

    def _write_number(self, k, v, dtype=None, unit=None):
        if dtype is None:
            if hasattr(v, "dtype"):
                dtype = v.dtype
            elif isinstance(v, float):
                dtype = "f8"
            elif isinstance(v, int):
                dtype = "i8"
            else:
                raise ValueError(f"Could not infer dtype for {type(v)}")

        dset = self._group.require_dataset(k, data=v, dtype=dtype, shape=())
        if unit:
            dset.attrs["unit"] = unit

    def _write_array(self, k, v, dtype=None, columns=None, units=None):
        if dtype is None:
            dtype = v.dtype

        if v.size > 16:
            opts = self._compression
        else:
            opts = {}
        dset = self._group.require_dataset(k, data=v, dtype=dtype,
                                           shape=v.shape, **opts)
        if columns:
            dset.attrs["columns"] = columns
        if units:
            dset.attrs["units"] = units

        return dset

    @staticmethod
    def _check_columns(v, columns):
        n = len(v)
        if columns and (len(columns) != n):
            raise ValueError(f"Invalid number of columns (expected {n} got "
                             f"{len(columns)})")

    @staticmethod
    def _check_units(v, units):
        n = len(v)
        if units and (len(units) != n):
            raise ValueError(f"Invalid number of units (expected {n} got "
                             f"{len(units)})")
    @staticmethod
    def _get_unit(v):
        try:
            return v.unit.name
        except AttributeError:
            return ""

    @property
    def elements(self):
        return ElementsIterator(self)

    @property
    def parent(self):
        return DataNode(self._group.parent)

    @property
    def children(self):
        return [node for node in self]

    @property
    def name(self):
        if self._name is None:
            self._name = os.path.basename(self._group.name)
        return self._name

    @property
    def path(self):
        return self._group.name

    @property
    def filename(self):
        return self._group.file.filename


class ClosingDataNode(DataNode):
    def __enter__(self) -> ClosingDataNode:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def open(file, mode="r"):
    f = _File(file, mode)
    return ClosingDataNode(f["/"])
