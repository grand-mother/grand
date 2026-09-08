# -*- coding: utf-8 -*-
r"""The two pure helpers in ``granddb/rootdblib.py``.

``sanitize_name`` and ``timestamp_to_date_time`` take a value and return a
value.  They are the only functions in granddb that can be tested without
building anything, and both feed names and dates into the catalogue, where a
change of format is not obviously wrong until something downstream cannot parse
it.
"""
import time

from granddb.rootdblib import sanitize_name, timestamp_to_date_time


def test_sanitize_name_removes_spaces_and_underscores():
    r"""Names lose whitespace and underscores, and keep everything else.

    Runs of either collapse to nothing rather than to a single character, which
    is what the regexp does and is worth stating: ``a__b`` and ``a_b`` both
    become ``ab``, so two distinct names can sanitize to the same one.
    """
    assert sanitize_name("Xiao Du Shan") == "XiaoDuShan"
    assert sanitize_name("gp_13") == "gp13"
    assert sanitize_name("a__b") == "ab"
    assert sanitize_name("a_b") == "ab", (
        "runs collapse to nothing, so a_b and a__b are indistinguishable after")
    assert sanitize_name("GP300") == "GP300", "digits and case are left alone"
    assert sanitize_name("") == ""


def test_timestamp_to_date_time_splits_a_unix_time_into_date_and_time():
    r"""A POSIX timestamp becomes ``("YYYYMMDD", "HHMMSS")``, in UTC.

    The conversion uses ``time.gmtime``, so it does not depend on the machine's
    zone -- which matters, since these strings end up in file names and database
    columns that are compared across sites.
    """
    # 2022-10-26 00:00:00 UTC, the run date of the Xiaodushan fixtures.
    stamp = 1666742400
    assert timestamp_to_date_time(stamp) == ("20221026", "000000")

    # A string is accepted too: the callers pass values straight out of a tree.
    assert timestamp_to_date_time(str(stamp)) == ("20221026", "000000")


def test_timestamp_to_date_time_is_utc_not_local():
    r"""Pinned separately, because a switch to ``localtime`` would still pass
    the test above on a machine running UTC."""
    stamp = 1666742400
    date, clock = timestamp_to_date_time(stamp)
    expected = time.strftime("%Y%m%d", time.gmtime(stamp))
    assert date == expected
    assert (date, clock) != (time.strftime("%Y%m%d", time.localtime(stamp)),
                             time.strftime("%H%M%S", time.localtime(stamp))) or \
        time.gmtime(stamp) == time.localtime(stamp), (
            "the conversion followed local time on a machine that is not UTC")
