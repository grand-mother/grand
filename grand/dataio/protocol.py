import ssl
import urllib.request
import zlib
import logging

logger = logging.getLogger(__name__)


def _disable_certs() -> None:
    """Disable certificates check"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


_disable_certs()


class InvalidBLOB(IOError):
    """Wrapper for store errors."""

    pass


def get(name: str, tag: str = "101") -> bytes:
    """Get a BLOB from the store.

    Parameters
    ----------
    name : str
        File to fetch.
    tag : str, optional
        Release tag to fetch it from.

    Returns
    -------
    bytes
        The downloaded contents.
    """
    # None, or an empty name, became a request for ".../None.gz" (#267)
    if not isinstance(name, str) or not name:
        raise TypeError("GRANDlib: protocol.get: 'name' must be a non-empty file name, got %r" % (name,))
    base = "https://github.com/grand-mother/store/releases/download"
    url = f"{base}/{tag}/{name}.gz"
    try:
        with urllib.request.urlopen(url) as f:
            return zlib.decompress(f.read(), wbits=31)
    except Exception as e:
        raise InvalidBLOB(e) from None
