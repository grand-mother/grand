from __future__ import annotations

from collections import OrderedDict
import os
from typing import Any, Optional, Tuple, Union

import astropy.units as u
from astropy.coordinates import BaseCoordinateFrame, BaseRepresentation,       \
                                CartesianRepresentation
from astropy.coordinates.representation import REPRESENTATION_CLASSES
from astropy.time import Time
import h5py
from h5py import Dataset as _Dataset, File as _File, Group as _Group
import numpy

from . import ECEF, LTP, Rotation

__all__ = ['DataNode', 'ElementsIterator']


class ElementsIterator:
    '''
    Iterator over the data elements of a node
    '''
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
    '''
    A node containing data elements and branches to sub-nodes
    '''

    _compression = {'compression': 'gzip', 'compression_opts': 9}

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
            self._write_string(k, v)
        elif isinstance(v, u.Quantity):
            self._write_quantity(k, v, dtype, unit)
        elif isinstance(v, (list, tuple)):
            self._write_table(k, v, dtype, columns, units)
        elif isinstance(v, numpy.ndarray):
            if columns:
                self._check_columns(v, columns)
            if units:
                self._check_units(v, units)
            self._write_array(k, v, dtype, columns, units)
        elif isinstance(v, BaseRepresentation):
            self._write_representation(k, v, dtype, unit, columns, units)
        elif isinstance(v, BaseCoordinateFrame):
            self._write_frame(k, v)
        else:
            self._write_number(k, v, dtype, unit)

    def _unpack(self, dset: _Dataset,
                      dtype: Union[numpy.DataType, str, None]=None) -> Any:
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
            metatype = dset.attrs['metatype']
        except KeyError:
            metatype = None

        if metatype is None:
            return v
        elif metatype == 'quantity':
            return self._unpack_quantity(dset, v)
        elif metatype == 'table':
            return self._unpack_table(dset, v)
        elif metatype.startswith('representation'):
            return self._unpack_representation(dset, v)
        elif metatype.startswith('frame'):
            return self._unpack_frame(dset, v)
        else:
            raise ValueError(f'Invalid metatype {metatype}')

        return v

    def _write_string(self, k, v, columns=None, units=None) -> _Dataset:
        if hasattr(v, 'encode'):
            encoding = 'UTF-8'
            v = v.encode()
        else:
            encoding = 'ASCII'

        v = numpy.bytes_(v)

        if len(v) > 128:
            opts = self._compression
        else:
            opts = {}
        dset = self._group.require_dataset(k, data=v, shape=v.shape,
                                           dtype=v.dtype, **opts)
        dset.attrs['encoding'] = encoding

        return dset

    @staticmethod
    def _unpack_string(dset, v):
        encoding = dset.attrs['encoding']
        if encoding != 'ASCII':
            return v.decode(encoding)
        else:
            return v.tobytes()

    def _write_quantity(self, k, v, dtype=None, unit=None) -> _Dataset:
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
        dset.attrs['unit'] = unit
        dset.attrs['metatype'] = 'quantity'

        return dset

    @staticmethod
    def _unpack_quantity(dset, v):
        unit = dset.attrs['unit']
        return v * u.Unit(unit)

    def _write_representation(self, k, v, dtype=None, unit=None,
        columns=None, units=None) -> _Dataset:

        components = v.components
        values = [getattr(v, f'_{name}') for name in components]

        if columns is None:
            columns = components

        if unit is not None:
            units = (unit, unit, unit)

        dset = self._write_table(k, values, dtype, columns, units)
        dset.attrs['metatype'] = f'representation/{v.get_name()}'

        return dset

    @staticmethod
    def _unpack_representation(dset, v):
        name = os.path.basename(dset.attrs['metatype'])
        cls = REPRESENTATION_CLASSES[name]
        units = dset.attrs['units']
        v = [v[i] * u.Unit(ui) for i, ui in enumerate(units)]
        return cls(*v)

    def _write_table(self, k, v, dtype=None, columns=None, units=None)         \
        -> _Dataset:

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
            m = 0
            data = numpy.zeros(n, dtype=dtype)
        else:
            data = numpy.zeros((n, m), dtype=dtype)

        for i, ui in enumerate(units):
            s = (i, slice(None)) if (m > 0) else i
            if ui:
                data[s] = v[i].to_value(ui)
            else:
                data[s] = v[i]

        dset = self._write_array(k, data, dtype, columns, units)
        dset.attrs['metatype'] = 'table'

        return dset

    @staticmethod
    def _unpack_table(dset, v):
        units = dset.attrs['units']
        try:
            m = dset.shape[1]
        except IndexError:
            m = 0

        table = []
        for i, ui in enumerate(units):
            s = (i, slice(None)) if (m > 0) else i
            vi = v[s]
            if ui:
                vi *= u.Unit(ui)
            table.append(vi)
        return table

    def _write_number(self, k, v, dtype=None, unit=None) -> _Dataset:
        if dtype is None:
            if hasattr(v, 'dtype'):
                dtype = v.dtype
            elif isinstance(v, float):
                dtype = 'f8'
            elif isinstance(v, int):
                dtype = 'i8'
            else:
                raise ValueError(f'Could not infer dtype for {type(v)}')

        dset = self._group.require_dataset(k, data=v, dtype=dtype, shape=())
        if unit:
            dset.attrs['unit'] = unit

    def _write_array(self, k, v, dtype=None, columns=None, units=None)         \
        -> _Dataset:

        if dtype is None:
            dtype = v.dtype

        if v.size > 16:
            opts = self._compression
        else:
            opts = {}
        dset = self._group.require_dataset(k, data=v, dtype=dtype,
                                           shape=v.shape, **opts)
        if columns:
            dset.attrs['columns'] = columns
        if units:
            dset.attrs['units'] = units

        return dset

    def _write_frame(self, k, v):
        if isinstance(v, ECEF):
            dset = self._group.require_dataset(k, data=None, shape=(),
                                               dtype='f8')
            dset.attrs['metatype'] = 'frame/ecef'
        elif isinstance(v, LTP):
            data = numpy.empty((4, 3))
            c = v._origin.represent_as(CartesianRepresentation)
            data[0,0] = c.x.to_value('m')
            data[0,1] = c.y.to_value('m')
            data[0,2] = c.z.to_value('m')
            data[1:,:] = v._basis

            dset = self._group.require_dataset(k, data=data, shape=data.shape,
                                               dtype=data.dtype)
            dset.attrs['metatype'] = 'frame/ltp'
            dset.attrs['orientation'] = v.orientation
            dset.attrs['magnetic'] = v.magnetic
            if v.declination is not None:
                dset.attrs['declination'] = v.declination.to_value('deg')
            if v.rotation is not None:
                dset.attrs['rotation'] = v.rotation.matrix
        else:
            raise NotImplementedError(type(v))

        if v.obstime is not None:
            dset.attrs['obstime'] = v.obstime.jd

    @staticmethod
    def _unpack_frame(dset, v):
        try:
            obstime = dset.attrs['obstime']
        except KeyError:
            obstime = None
        else:
            obstime = Time(obstime, format='jd')

        name = os.path.basename(dset.attrs['metatype'])
        if name == 'ecef':
            return ECEF(obstime=obstime)
        elif name == 'ltp':
            location = ECEF(dset[0,:] * u.m)

            try:
                rotation = dset.attrs['rotation']
            except KeyError:
                rotation = None
            else:
                rotation = Rotation.from_matrix(rotation)

            try:
                declination = dset.attrs['declination'] << u.deg
            except KeyError:
                declination = None

            return LTP(location=location,
                       orientation=dset.attrs['orientation'],
                       magnetic=dset.attrs['magnetic'],
                       declination=declination,
                       obstime=obstime, rotation=rotation)
        else:
            raise NotImplementedError(name)

    @staticmethod
    def _check_columns(v, columns):
        n = len(v)
        if columns and (len(columns) != n):
            raise ValueError(f'Invalid number of columns (expected {n} got '
                             f'{len(columns)})')

    @staticmethod
    def _check_units(v, units):
        n = len(v)
        if units and (len(units) != n):
            raise ValueError(f'Invalid number of units (expected {n} got '
                             f'{len(units)})')
    @staticmethod
    def _get_unit(v):
        try:
            return v.unit.name
        except AttributeError:
            return ''

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


def open(file, mode='r'):
    f = _File(file, mode)
    return ClosingDataNode(f['/'])
