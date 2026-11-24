import numpy as np

from grand.basis.du_network import DetectorUnitNetwork


def test_init_pos_id():
    dun = DetectorUnitNetwork()
    du_pos = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
    du_id = np.arange(2) + 20
    dun.init_pos_id(du_pos, du_id)
    assert np.allclose(dun.du_pos, du_pos)


def test_reduce_nb_du():
    dun = DetectorUnitNetwork()
    du_pos = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
    du_id = np.arange(2) + 20
    dun.init_pos_id(du_pos, du_id)
    new_nb = 1
    assert dun.du_pos.shape[0] > new_nb
    dun.reduce_nb_du(new_nb)
    assert dun.du_pos.shape[0] == new_nb


def test_get_sub_network():
    dun = DetectorUnitNetwork()
    du_pos = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
    du_id = np.arange(2) + 20
    dun.init_pos_id(du_pos, du_id)
    subnet = dun.get_sub_network([0, 1])  # list of selected DUs.
    assert subnet.du_pos.shape[0] == 2


def test_get_surface():
    dun = DetectorUnitNetwork()
    du_pos = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]])
    dun.init_pos_id(du_pos)
    area_km2 = dun.get_surface()
    assert area_km2 == 0.5
    du_pos = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 0]])
    dun.init_pos_id(du_pos)
    area_km2 = dun.get_surface()
    assert area_km2 == 1
    du_pos = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 0], [2000, 0, 0]])
    dun.init_pos_id(du_pos)
    area_km2 = dun.get_surface()
    assert area_km2 == 1.5


def test_keep_only_du_with_index():
    dun = DetectorUnitNetwork()
    du_pos = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 0], [2000, 0, 0]])
    dun.init_pos_id(du_pos)
    dun.keep_only_du_with_index([0, 2, 4])
    assert dun.get_nb_du() == 3
    area_km2 = dun.get_surface()
    assert area_km2 == 1
