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
