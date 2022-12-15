import grand.manage_log as mlg
from grand.geo.coordinates import geoid_undulation


def test_check_logger_level():
    ret_lev = mlg._check_logger_level("info")
    assert ret_lev == mlg.logging.INFO
    ret_lev = mlg._check_logger_level("info2")
    assert ret_lev == mlg.logging.DEBUG


def test_get_string_now():
    ret = mlg._get_string_now()
    assert ret.find("T") == 10


def test_get_logger_path():
    ret = mlg._get_logger_path("/toto/grand/tutu.py")
    assert ret == "grand.tutu"
    ret = mlg._get_logger_path("/toto/gran/tutu.py")
    assert ret == "toto.gran.tutu"


def test_get_logger_for_script():
    logger = mlg.get_logger_for_script(__file__)
    r_log = mlg._get_logger_path(__file__)
    fn_log = "tests/test_log.txt"
    mlg.create_output_for_logger(log_file=fn_log, log_stdout=False)
    logger.info("test")
    logger.info(mlg.string_begin_script())
    logger.info(mlg.chrono_start())
    logger.info(mlg.chrono_string_duration())
    logger.info(mlg.string_end_script())
    mlg.close_output_for_logger()
    with open(fn_log, "r") as flog:
        all_log = flog.read()
        assert all_log.find(r_log) > 0
        assert all_log.find("Begin") > 0
        assert all_log.find("End at") > 0
        assert all_log.find("Chrono start") > 0
        assert all_log.find("Chrono duration") > 0


def test_get_logger_for_script_out_pkg():
    p_script = "/home/user/test/script.py"
    logger = mlg.get_logger_for_script(p_script)
    r_log = "home.user.test.script"
    fn_log = "tests/test_log_out_pkg.txt"
    mlg.create_output_for_logger(log_file=fn_log, log_stdout=False)
    logger.info("test")
    logger.info(mlg.string_begin_script())
    logger.info(mlg.chrono_start())
    geoid_undulation()
    logger.info(mlg.chrono_string_duration())
    logger.info(mlg.string_end_script())
    mlg.close_output_for_logger()
    with open(fn_log, "r") as flog:
        all_log = flog.read()
        assert all_log.find(r_log) > 0
        assert all_log.find("Begin") > 0
        assert all_log.find("End at") > 0
        assert all_log.find("Chrono start") > 0
        assert all_log.find("Chrono duration") > 0
