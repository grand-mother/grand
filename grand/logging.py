import logging
from logging import DEBUG, INFO, WARNING, ERROR
from typing import cast, Optional
import sys

__all__ = ['getLogger', 'Logger', 'StreamHandler']


class StreamHandler(logging.Handler):
    '''
    A specialised Handler that logs messages below INFO to stdout and above to
    stderr
    '''

    terminator = '\n'

    def flush(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno <= logging.INFO:
            stream = sys.stdout
        else:
            stream = sys.stderr

        try:
            msg:str = self.format(record)
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            stream.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)


class Logger(logging.getLoggerClass()): # type: ignore
    '''
    A specialised Logger with predefined usage. The stream and file attributes
    give access to the corresponding handlers. The path attributes allows to set
    the optional output file.
    '''
    _initial_level = logging.WARNING

    def reset(self) -> None:
        self._default_level = self._initial_level
        super().setLevel(self._initial_level)

        # Print formatter
        self._formatter = logging.Formatter(
            '%(asctime)s [ %(levelname)-7s ] (%(name)s) %(message)s')

        # Standard output
        try:
            self.removeHandler(self.stream) # type: ignore
        except AttributeError:
            pass

        self.stream = StreamHandler()
        self.stream.setLevel(self._initial_level)
        self.stream.setFormatter(self._formatter)
        self.addHandler(self.stream)

        # File output
        try:
            self.removeHandler(self.file)
        except AttributeError:
            pass

        self._path: Optional[str] = None
        self.file: Optional[logging.FileHandler] = None

    def setLevel(self, level):
        self._default_level = level
        super().setLevel(level)
        self.stream.setLevel(level)
        if self.file is not None:
            self.file.setLevel(level)

    @property
    def path(self) -> Optional[str]:
        return self._path

    @path.setter
    def path(self, path: Optional[str]) -> None:
        self.logger.removeHandler(self.file)
        if path is not None:
            self.file = logging.FileHandler(path)
            self.file.setLevel(self._default_level)
            self.file.setFormatter(self._formatter)
            self.logger.addHandler(self.file)


def getLogger(name: str) -> Logger: # XXX make this patchable for external users
    '''
    Get a specialised logger for the given namespace
    '''
    base = logging.getLoggerClass()
    logging.setLoggerClass(Logger)

    try:
        logger = cast(Logger, logging.getLogger(name))
        logger.reset()
    finally:
        logging.setLoggerClass(base)

    return logger
