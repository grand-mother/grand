import os
import grand.manage_log as mlg
from granddb.datamanager import DataManager
import argparse
logger = mlg.get_logger_for_script(__name__)


def main():
    r"""Registers each path named on the command line, file or directory.

    Unlike its siblings this one dispatches on what the path is, so it
    takes either.

    Called by the ``__main__`` guard below.  The body used to run at import,
    which meant that merely importing this module parsed ``sys.argv`` and
    built a :class:`DataManager` -- opening the configured database.
    """
    mlg.create_output_for_logger("debug", log_stdout=True)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", default="config.ini", help="Config file to use")
    argParser.add_argument("-r", "--repository", default="", help="Repository")
    argParser.add_argument('files', nargs='+', default=[], help='Files or dir to register')
    args = argParser.parse_args()

    # if config is given as absolute path, use it. If not then use path relative to script
    if args.config[0] == '/':
        config_path = args.config
    else:
        config_path = os.path.dirname(__file__)+"/"+args.config

    dm = DataManager(config_path)
    if args.repository == '':
        repo_name = None
    else:
        repo_name = args.repository
    for file in args.files:
        try:
            if os.path.isfile(file):
                logger.info(f'Register file ${file}')
                dm.register_file(localfile=file,  repository=repo_name, again=True)
            elif os.path.isdir(file):
                logger.info(f'Register directory ${file}')
                dm.register_dataset(directory=file, repository=repo_name, again=True)
        except Exception as e:
            logger.error(f'Error when importing {file}. Skipping.')
            logger.error(f'Error was {e}.')


if __name__ == "__main__":
    main()
