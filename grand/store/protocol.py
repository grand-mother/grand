import ssl
import urllib.request
import zlib

def _disable_certs() -> None:
    '''Disable certificates check'''
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

_disable_certs()


class InvalidBLOB(IOError):
    '''Wrapper for store errors.'''
    pass


def get(name: str, tag: str='101') -> bytes:
    '''Get a BLOB from the store.
    '''
    base = 'https://github.com/grand-mother/store/releases/download'
    url = f'{base}/{tag}/{name}.gz'
    try:
        with urllib.request.urlopen(url) as f:
            return zlib.decompress(f.read(), wbits=31)
    except Exception as e:
        raise InvalidBLOB(e) from None
