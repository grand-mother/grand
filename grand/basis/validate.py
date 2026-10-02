# -*- coding: utf-8 -*-
r"""Input checks for GRANDlib's public functions, with messages that help.

Every check raises a standard exception -- ``TypeError`` for the wrong kind
of value, ``ValueError`` for a value of the right kind but out of range or
of the wrong shape, ``FileNotFoundError`` for a missing file -- so callers
can catch them as usual.  The message starts with ``GRANDlib:``, names the
function and the argument, says what was expected and shows what was given::

    ValueError: GRANDlib: Geodetic: 'latitude' must be between -90 and 90 degrees, got 200.0

Warnings use :class:`GRANDlibWarning`, so they can be filtered as a group::

    import warnings
    from grand.basis.validate import GRANDlibWarning
    warnings.simplefilter("error", GRANDlibWarning)   # make them errors

Checks return the value converted to the form the code needs (a float, an
int, a NumPy array), so a function can validate and convert in one line.
"""

import numbers
import os
import warnings

import numpy as np

__all__ = [
    "GRANDlibWarning",
    "warn",
    "message",
    "as_real",
    "as_integer",
    "as_array",
    "in_range",
    "positive",
    "non_negative",
    "nonzero",
    "one_of",
    "same_length",
    "existing_file",
    "existing_directory",
    "coerce_to_dtype",
]


class GRANDlibWarning(UserWarning):
    """Warnings issued by GRANDlib: suspicious input that is still usable."""


def message(where, text):
    r"""Formats an error or warning message: ``"GRANDlib: <where>: <text>"``.

    Parameters
    ----------
    where : str
        The function, class or field concerned, for example ``"Geodetic"``.
    text : str
        What is wrong, and what was given.

    Returns
    -------
    str
    """
    return "GRANDlib: %s: %s" % (where, text) if where else "GRANDlib: %s" % text


def warn(where, text, stacklevel=3):
    r"""Issues a :class:`GRANDlibWarning` pointing at the caller's code.

    Parameters
    ----------
    where : str
        The function, class or field concerned.
    text : str
        What is suspicious, and what will happen.
    stacklevel : int, optional
        How far up the stack the warning points; the default points at the
        code that called the function doing the check.
    """
    warnings.warn(message(where, text), GRANDlibWarning, stacklevel=stacklevel)


def _show(value):
    r"""A short representation of `value` for messages."""
    if isinstance(value, np.ndarray):
        if value.size <= 6:
            return np.array2string(value, separator=", ")
        return "an array of shape %s" % (value.shape,)
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _is_real(value):
    return (isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))) \
        or (isinstance(value, np.ndarray) and value.ndim == 0
            and value.dtype.kind in "iuf")


def as_real(value, name, where, finite=True):
    r"""Checks `value` is a real number (not a string, not complex) and returns it as a float.

    Parameters
    ----------
    value : object
        The value given.
    name : str
        The argument's name, for the message.
    where : str
        The function concerned, for the message.
    finite : bool, optional
        Also refuse NaN and infinity.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        If `value` is not a real number.
    ValueError
        If `finite` and `value` is NaN or infinite.
    """
    if not _is_real(value):
        raise TypeError(message(where, "'%s' must be a real number, got %s (%s)"
                                % (name, _show(value), type(value).__name__)))
    value = float(value)
    if finite and not np.isfinite(value):
        raise ValueError(message(where, "'%s' must be finite, got %s" % (name, value)))
    return value


def as_integer(value, name, where, minimum=None, maximum=None):
    r"""Checks `value` is a whole number, optionally within limits, and returns it as an int.

    A float with no fractional part (``3.0``) is accepted; ``1.7`` is not, nor
    is a boolean.

    Parameters
    ----------
    value : object
        The value given.
    name : str
        The argument's name, for the message.
    where : str
        The function concerned, for the message.
    minimum, maximum : int, optional
        Inclusive limits.

    Returns
    -------
    int

    Raises
    ------
    TypeError
        If `value` is not a number, or has a fractional part.
    ValueError
        If it lies outside the limits.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(message(where, "'%s' must be an integer, got the boolean %s" % (name, value)))
    if isinstance(value, numbers.Integral):
        result = int(value)
    elif _is_real(value) and np.isfinite(float(value)) and float(value).is_integer():
        result = int(value)
    elif _is_real(value):
        raise TypeError(message(where, "'%s' must be an integer, got %s" % (name, _show(value))))
    else:
        raise TypeError(message(where, "'%s' must be an integer, got %s (%s)"
                                % (name, _show(value), type(value).__name__)))
    if (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise ValueError(message(where, "'%s' must be an integer %s, got %d"
                                 % (name, _limits(minimum, maximum), result)))
    return result


def _limits(minimum, maximum, unit=""):
    unit = " " + unit if unit else ""
    if minimum is not None and maximum is not None:
        return "between %s and %s%s" % (minimum, maximum, unit)
    if minimum is not None:
        return ">= %s%s" % (minimum, unit)
    return "<= %s%s" % (maximum, unit)


def as_array(value, name, where, ndim=None, shape=None, min_length=None, finite=False, dtype=float):
    r"""Converts `value` to a NumPy array and checks its dimensions and contents.

    Parameters
    ----------
    value : array-like
        The value given.
    name : str
        The argument's name, for the message.
    where : str
        The function concerned, for the message.
    ndim : int, optional
        Required number of dimensions.
    shape : tuple, optional
        Required shape; ``None`` in it matches any length, so ``(None, 3)``
        means "any number of rows, three columns".
    min_length : int, optional
        Minimum length of the first axis.
    finite : bool, optional
        Refuse NaN and infinity.
    dtype : type, optional
        The dtype to convert to; ``float`` by default.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    TypeError
        If `value` cannot be converted to numbers.
    ValueError
        If its shape, length or contents are wrong.
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(message(where, "'%s' must be an array of numbers, got the string %s"
                                % (name, _show(value))))
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(message(where, "'%s' must be an array of numbers, got %s (%s)"
                                % (name, _show(value), error))) from None
    if shape is not None:
        ndim = len(shape)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(message(where, "'%s' must have %d dimension%s%s, got shape %s"
                                 % (name, ndim, "" if ndim == 1 else "s",
                                    " (shape %s)" % _shape(shape) if shape else "", array.shape)))
    if shape is not None and any(want is not None and got != want
                                 for want, got in zip(shape, array.shape)):
        raise ValueError(message(where, "'%s' must have shape %s, got %s"
                                 % (name, _shape(shape), array.shape)))
    if min_length is not None and (array.ndim == 0 or len(array) < min_length):
        raise ValueError(message(where, "'%s' must have at least %d element%s, got %d"
                                 % (name, min_length, "" if min_length == 1 else "s",
                                    0 if array.ndim == 0 else len(array))))
    if finite and array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(message(where, "'%s' must not contain NaN or infinity (%d of %d values are)"
                                 % (name, np.count_nonzero(~np.isfinite(array)), array.size)))
    return array


def _shape(shape):
    parts = ["N" if n is None else str(n) for n in shape]
    return "(%s%s)" % (", ".join(parts), "," if len(parts) == 1 else "")


def in_range(value, name, where, minimum=None, maximum=None, unit=""):
    r"""Checks every element of `value` lies within ``[minimum, maximum]``.

    Parameters
    ----------
    value : float or numpy.ndarray
        The value given, already numeric.  NaN passes; use `finite` checks to
        refuse it.
    name, where : str
        For the message.
    minimum, maximum : float, optional
        Inclusive limits.
    unit : str, optional
        The unit, for the message.

    Returns
    -------
    float or numpy.ndarray
        `value`, unchanged.

    Raises
    ------
    ValueError
        If any element lies outside the limits.
    """
    array = np.asarray(value)
    bad = np.zeros(array.shape, bool)
    if minimum is not None:
        bad |= array < minimum
    if maximum is not None:
        bad |= array > maximum
    if np.any(bad):
        shown = array[bad].ravel()[0] if array.ndim else array.item()
        raise ValueError(message(where, "'%s' must be %s, got %s"
                                 % (name, _limits(minimum, maximum, unit), shown)))
    return value


def positive(value, name, where, unit=""):
    r"""Checks every element of `value` is strictly positive; returns it unchanged.

    Raises
    ------
    ValueError
    """
    array = np.asarray(value)
    if np.any(array <= 0):
        raise ValueError(message(where, "'%s' must be positive%s, got %s"
                                 % (name, " (%s)" % unit if unit else "",
                                    array[array <= 0].ravel()[0] if array.ndim else array.item())))
    return value


def non_negative(value, name, where, unit=""):
    r"""Checks no element of `value` is negative; returns it unchanged.

    Raises
    ------
    ValueError
    """
    array = np.asarray(value)
    if np.any(array < 0):
        raise ValueError(message(where, "'%s' must not be negative%s, got %s"
                                 % (name, " (%s)" % unit if unit else "",
                                    array[array < 0].ravel()[0] if array.ndim else array.item())))
    return value


def nonzero(value, name, where):
    r"""Checks no element of `value` is zero; returns it unchanged.

    Raises
    ------
    ValueError
    """
    if np.any(np.asarray(value) == 0):
        raise ValueError(message(where, "'%s' must not be zero" % name))
    return value


def one_of(value, choices, name, where):
    r"""Checks `value` is one of `choices`; returns it unchanged.

    Raises
    ------
    ValueError
        Listing the accepted values.
    """
    if value not in choices:
        raise ValueError(message(where, "'%s' must be one of %s, got %s"
                                 % (name, ", ".join(repr(c) for c in choices), _show(value))))
    return value


def same_length(where, **arrays):
    r"""Checks the named arrays all have the same length (first axis).

    Parameters
    ----------
    where : str
        For the message.
    **arrays : array-like
        The arrays, by argument name.

    Raises
    ------
    ValueError
        Giving each array's length.
    """
    lengths = {name: len(np.atleast_1d(array)) for name, array in arrays.items()}
    if len(set(lengths.values())) > 1:
        names = ["'%s'" % n for n in lengths]
        listed = ", ".join(names[:-1]) + " and " + names[-1]
        raise ValueError(message(where, "%s must have the same length, got %s"
                                 % (listed, ", ".join("%d" % n for n in lengths.values()))))


def existing_file(path, name, where):
    r"""Checks `path` names an existing file; returns it unchanged.

    Raises
    ------
    TypeError
        If `path` is not a string or path.
    FileNotFoundError
        If there is no such file.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(message(where, "'%s' must be a file name, got %s (%s)"
                                % (name, _show(path), type(path).__name__)))
    if not os.path.isfile(path):
        if os.path.isdir(path):
            raise FileNotFoundError(message(where, "'%s' must be a file, but %s is a directory"
                                            % (name, os.fspath(path))))
        raise FileNotFoundError(message(where, "no such file: %s" % os.fspath(path)))
    return path


def existing_directory(path, name, where):
    r"""Checks `path` names an existing directory; returns it unchanged.

    Raises
    ------
    TypeError
        If `path` is not a string or path.
    FileNotFoundError
        If there is no such directory.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(message(where, "'%s' must be a directory name, got %s (%s)"
                                % (name, _show(path), type(path).__name__)))
    if not os.path.isdir(path):
        raise FileNotFoundError(message(where, "no such directory: %s" % os.fspath(path)))
    return path


def coerce_to_dtype(value, dtype, where):
    r"""Converts `value` for storage as `dtype`, refusing silent changes.

    Used by the data-tree fields.  For an integer dtype, the value must be a
    whole number within the dtype's range (``1.7`` and ``-1`` for an
    unsigned field are refused, rather than stored as 1 and 4294967295);
    for a float dtype, a real number.  Text that reads as a number (``"1618"``)
    is converted, with a :class:`GRANDlibWarning`, as it always was; other
    text is refused.  Works on scalars and arrays.

    Parameters
    ----------
    value : object
        The value given.
    dtype : numpy.dtype or type
        The storage type.
    where : str
        The field, for example ``"TRun.run_number"``.

    Returns
    -------
    object
        `value` as a NumPy array of `dtype` (0-d for a scalar).

    Raises
    ------
    TypeError
        If `value` is not numeric, or not whole for an integer field.
    ValueError
        If it is out of the dtype's range.
    """
    dtype = np.dtype(dtype)
    if dtype.kind not in "iuf":
        return np.asarray(value, dtype=dtype)
    # None was stored as NaN in a float field, and called "NaN or infinity"
    # in an integer one (#206)
    if value is None:
        raise TypeError(message(where, "must be a number, got None; use NaN for an "
                                "unknown value of a float field"))
    try:
        given = np.asarray(value)
    except (TypeError, ValueError):
        raise TypeError(message(where, "must be a number, got %s" % _show(value))) from None
    if given.dtype.kind in "USO":
        # Text that reads as a number ("1618", from a file name or the command
        # line) was always stored as that number: keep converting it, but say
        # so, so the caller can pass the number itself.  Other text is refused.
        has_text = any(isinstance(v, (str, bytes)) for v in given.ravel())
        try:
            given = given.astype(float)
        except (TypeError, ValueError):
            raise TypeError(message(where, "must be a number, got %s" % _show(value))) from None
        if has_text:
            warn(where, "got the text %s; stored as the number it reads as. Pass a number "
                 "instead" % _show(value), stacklevel=5)
    if given.dtype.kind == "c":
        raise TypeError(message(where, "must be a real number, got the complex %s" % _show(value)))
    if dtype.kind == "f":
        with np.errstate(over="ignore", under="ignore"):
            converted = given.astype(dtype)
        # 1e39 in a float32 field read back as inf, and 1e-46 as 0 (#289)
        finite = np.isfinite(given)
        if np.any(finite & ~np.isfinite(converted)):
            raise ValueError(message(where, "%s does not fit in %s (largest %g)"
                                     % (_show(value), dtype, np.finfo(dtype).max)))
        if np.any(finite & (given != 0) & (converted == 0)):
            warn(where, "%s is below the smallest %s and is stored as 0" % (_show(value), dtype),
                 stacklevel=5)
        return converted
    # Integer storage
    if given.dtype.kind == "b":
        raise TypeError(message(where, "must be an integer, got a boolean"))
    if given.dtype.kind == "f":
        if not np.all(np.isfinite(given)):
            raise ValueError(message(where, "must be an integer, got NaN or infinity"))
        if not np.all(np.equal(np.mod(given, 1), 0)):
            raise TypeError(message(where, "must be an integer, got %s" % _show(value)))
    info = np.iinfo(dtype)
    if given.size and (given.min() < info.min or given.max() > info.max):
        bad = given.min() if given.min() < info.min else given.max()
        raise ValueError(message(where, "must be an integer between %d and %d (stored as %s), got %s"
                                 % (info.min, info.max, dtype, int(bad) if float(bad).is_integer() else bad)))
    return given.astype(dtype)


#: Ranges outside which a value is almost certainly in the wrong unit (#266):
#: ``(low, high, unit, what a value outside usually is)``.
PLAUSIBLE = {
    "frequency_mhz": (0.0, 1e5, "MHz", "a frequency in Hz or GHz"),
    "sampling_rate_mhz": (1.0, 1e5, "MHz", "a rate in Hz or GHz"),
    "time_step_ns": (1e-3, 1e4, "ns", "a time step in seconds"),
    "angle_rad": (-2 * np.pi, 2 * np.pi, "rad", "an angle in degrees"),
}


def plausible(value, name, where, kind):
    r"""Warns if `value` lies outside the plausible range of its unit; returns it unchanged.

    Type and range checks cannot tell 500 MHz given as 500e6 from a real
    500e6 MHz; this catches the unit mistakes that silently give results
    wrong by orders of magnitude (#266), with a `GRANDlibWarning` rather than
    an error, since the ranges are generous but not physical limits.

    Parameters
    ----------
    value : float or array_like
        The value to look at; non-finite elements are ignored.
    name : str
        The argument's name.
    where : str
        The function it was given to.
    kind : str
        A key of :data:`PLAUSIBLE`.
    """
    low, high, unit, likely = PLAUSIBLE[kind]
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return value
    finite = array[np.isfinite(array)]
    if finite.size and (finite.min() < low or finite.max() > high):
        bad = finite[(finite < low) | (finite > high)].ravel()[0]
        warnings.warn(message(where, "'%s' = %g is outside %g to %g %s; is it %s?"
                              % (name, bad, low, high, unit, likely)),
                      GRANDlibWarning, stacklevel=3)
    return value
