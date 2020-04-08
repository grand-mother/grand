:mod:`~grand.io` --- Utilities for reading or writing data
==========================================================
.. module:: grand.io

----

Overview
--------

The :mod:`grand.io` module provides utilities for easy reading (writing) Python
objects from (to) data files. The following types are supported: numpy
:class:`~numpy.array`, :class:`~astropy.coordinates.BaseRepresentation`,
:class:`~bytes`, :class:`~float`, :class:`~int`, astropy
:class:`~astropy.units.Quantity`, :class:`~str`. In addition any :class:`~list`
or :class:`~tuple` of numpy :class:`~numpy.array` or astropy
:class:`~astropy.units.Quantity` are stored as a numeric table with annotated
columns.

Reading and writing coordinate frames is only supported for the :mod:`~grand`
specific :class:`~grand.ECEF` and :class:`grand.LTP` frames. Note that the
coordinates data, if any, are not stored. If needed they must written explictly
as a separate entry.

Inside a file data are structured under :class:`~grand.io.DataNode`.  The
:func:`~grand.io.open` function allows to access the root node of a file.
Conceptualy a :class:`~grand.io.DataNode` can be seen as a folder within a file
system while the data would be files inside the folder.

.. note::

   A C complient memory layout is used for storing the data allowing for an
   efficient read back from C. Numeric values are annotated, e.g. with a unit,
   column name or a metatype. Note also that Python objects are preserved when
   writing and reading back from a data file, e.g. the object type is restored.

.. warning::

   The HDF5 format is currently used since it allows a hierarchical organization
   of data, has bindings both for C and Python and automatic compression. Note
   however that several `issues`_ have been reported when using HDF5, e.g.
   reliability and performances. Therefore, the underlying data format might
   change in the future, e.g. for a tar archive which actually provides the
   same features.

Accessing data files
--------------------

Data files are accessed using the :func:`~grand.io.open` function. The semantic
is the same than the Python :func:`~open` or C `fopen` functions.

.. autofunction:: grand.io.open

   Open file and return the root :class:`~grand.io.DataNode` object. If the file
   cannot be opened, an :class:`~OSError` is raised.

   For example, the following creates a new data file using the root
   :class:`~grand.io.DataNode` as a closing context manager.

   >>> with io.open('data.hdf5', 'w') as root:
   ...     pass

   In order to read from (append to) a file use the `'r'` (`'a'`) mode.


Managing data nodes
-------------------

.. autoclass:: grand.io.DataNode

   Sub-nodes can be accessed by index providing their relative path w.r.t. this
   node. For example the following gets a reference to the sub-node named
   *apples*.

   ..
      >>> root = io.open('data.hdf5', 'w')
      >>> node = root.branch('apples')

   >>> node = root['apples']

   ..
      >>> root.close()

   Note that an :class:`~IndexError` is raised if the sub-node does not exist.
   Use the :func:`~grand.io.DataNode.branch` method in order to create a new
   sub-node.

   The :func:`~grand.io.DataNode.read` and :func:`~grand.io.DataNode.write`
   methods allow to read and write data to this node.

   .. autoproperty:: grand.io.DataNode.children

      Iterator over the sub-nodes inside this node.

      For example, the loop below iterates over all sub-nodes below the root
      one.

      ..
         >>> root = io.open('data.hdf5')

      >>> for node in root.children:
      ...     pass

      ..
         >>> root.close()

   .. autoproperty:: grand.io.DataNode.elements

      Iterator over the data elements inside the node.

      For example, the loop below iterates over all data elements in the root
      node.

      ..
         >>> root = io.open('data.hdf5')

      >>> for name, data in root.elements:
      ...     pass

      ..
         >>> root.close()

      .. note::

         The data are loaded from disk at each loop iteration. Use the
         :func:`~grand.io.DataNode.read` method instead if you only want to
         load a specific data element.

   .. autoproperty:: grand.io.DataNode.filename

      The name of the data file containing this :class:`~grand.io.DataNode`.

   .. autoproperty:: grand.io.DataNode.name

      The name of this :class:`~grand.io.DataNode`.

   .. autoproperty:: grand.io.DataNode.parent

      A reference to the parent :class:`~grand.io.DataNode` or `None`.

   .. autoproperty:: grand.io.DataNode.path

      The full path of this :class:`~grand.io.DataNode` w.r.t. the root node.

   .. automethod:: grand.io.DataNode.branch

      Get a reference to a sub-node.

      .. note::

         If the node does not exists it is created and initialised empty. Use an
         indexed access instead if you want to access only existing sub-nodes.

   .. automethod:: grand.io.DataNode.close

      Close the data file containing the current node.

      .. warning::

         Closing the data file disables all related nodes, parents and children,
         which might lead to unexpected results. Therefore it is stronly
         recommended to wrap all I/Os within a root node context (i.e. using a
         `with` statement as shown below) instead of explictly calling the
         :func:`~grand.io.DataNode.close` method.

         >>> with open('data.hdf5') as root:
         ...     # Do all I/Os within this context
         ...     pass

         The data file is automatically closed when exiting the root node's
         context. Note that only the root node is a closing context. Contexts
         spawned from a sub-node do not close the data file.

   .. automethod:: grand.io.DataNode.read

      Read data from this node.

      The optional argument *dtype* allows to specify the data type to use
      for the read values. By default the native data type in the file is used.

      Multiple data can be read at once by providing multiple arguments. For
      example the following reads out two data elements from the root node.

      ..
         >>> root = io.open('data.hdf5', 'w')
         >>> root.write('frequency', 1 * u.Hz)
         >>> root.write('position', CartesianRepresentation(1, 2, 3, unit='m'))

      >>> frequency, position = root.read('frequency', 'position')

      ..
         >>> root.close()

   .. automethod:: grand.io.DataNode.write

      Write data to this node.

      The data type (*dtype*) can be explictly specified a `numpy data type`_.
      For example the following writes an astropy
      :class:`~astropy.units.Quantity` as a 32 bits floating point.

      ..
         >>> root = io.open('data.hdf5', 'w')

      >>> root.write('frequency', 1 * u.Hz, dtype='f')

      ..
         >>> root.close()

      Note that if *dtype* is omitted the native Python precision is used when
      writing the data to file.

      The *unit* keyword allows to specify the unit to use when writing an
      astropy :class:`~astropy.units.Quantity`. If omitted the native unit is
      used.

      The *units* and *columns* keywords allow to specify the units an names of
      columns when writing a table, i.e. a :class:`~list` or :class:`~tuple` of
      numpy :class:`~numpy.array` or astropy :class:`~astropy.units.Quantity`.


Examples
--------

Serialising Python data
^^^^^^^^^^^^^^^^^^^^^^^

The following example shows how to write basic Python objects to a data file.

..
   >>> import numpy as np

>>> with io.open('data.hdf5', 'w') as root:
...     root.write('example_of_cstring', b'This is a C like string\x00')
...     root.write('example_of_str', 'This is a Python string')
...     root.write('example_of_number', 1)
...     root.write('example_of_array', np.array((1, 2, 3)))

.. note::

   Python :class:`~str` objects differ from C ones. In order to generate a C
   like string a :class:`bytes` object must be used with an explicit null
   termination.

Conversely, reading the data back can be done as following.

>>> with io.open('data.hdf5') as root:
...     cstring = root.read('example_of_cstring')
>>> python_string = cstring.decode()

Working with physical data
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example illustrates how to create a new data file and populate it
with some physical data organised under various branches.

>>> with io.open('data.hdf5', 'w') as root:
...     root.write('energy', 1E+18 * u.eV)
...
...     with root.branch('fields/a0') as a0:
...         r = CartesianRepresentation(0, 0, 0, unit='m')
...         a0.write('r', r, dtype='f') # Store the data with a specific format
...
...         E = CartesianRepresentation(
...             np.array((0, 0, 0)),
...             np.array((0, 1, 0)),
...             np.array((0, 0, 0)),
...             unit='uV/m'
...         )
...         a0.write('E', E, unit='V/m') # Store the data with a specific unit



.. _issues: https://cyrille.rossant.net/moving-away-hdf5
.. _numpy data type: https://numpy.org/devdocs/user/basics.types.html
