from granddb.datamanager import DataManager
import os
import argparse
import grand.manage_log as mlg
logger = mlg.get_logger_for_script(__name__)


def main():
    r"""Refreshes the database's materialized views.

    Reads the connection settings from the config file given with ``-c``,
    defaulting to ``config.ini`` beside this script.

    Called by the ``__main__`` guard below.  The body used to run at import,
    which meant that merely importing this module parsed ``sys.argv`` and
    built a :class:`DataManager` -- opening the configured database.
    """
    # WARNING is what this script effectively ran at before granddb's library
    # modules stopped configuring logging on its behalf.
    mlg.create_output_for_logger("warning", log_stdout=True)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config",default="config.ini", help="Config file to use")
    args = argParser.parse_args()

    if args.config[0] == '/':
        config_path = args.config
    else:
        config_path = os.path.dirname(__file__)+"/"+args.config

    dm = DataManager(config_path)

    materialized_views = ['datamat']
    for view in materialized_views:
        logger.info(f'refreshing {view}.')
        dm.database().execute_sql(str('refresh materialized view '+view))


if __name__ == "__main__":
    main()
