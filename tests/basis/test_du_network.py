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
    subnet = dun.get_sub_network([0,1]) # list of selected DUs.
    assert subnet.du_pos.shape[0] == 2

#def test_get_surface():
#    # RK: this is a lazy test. Improve it in the next release.
#    dun = DetectorUnitNetwork()
#    du_pos = np.arange(10 * 3, dtype=np.float32).reshape((10, 3))
#    du_id = np.arange(10) + 20
#    dun.init_pos_id(du_pos, du_id)
#    area_km2 = dun.get_surface()
#    assert area_km2 is not None