#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def enu_basis(lat_deg, lon_deg):
    """Returns the ECEF-frame E, N, U basis vectors at a location."""
    th, ph = np.radians(lat_deg), np.radians(lon_deg)
    E = np.array([-np.sin(ph),  np.cos(ph), 0.0])
    N = np.array([-np.sin(th)*np.cos(ph), -np.sin(th)*np.sin(ph), np.cos(th)])
    U = np.array([ np.cos(th)*np.cos(ph),  np.cos(th)*np.sin(ph), np.sin(th)])
    return np.vstack([E, N, U])

# The GRANDProto300 site at Dunhuang, from arXiv:2408.10926 Fig. 7.
R = enu_basis(40.98, 93.95)

print("R is orthogonal:", np.allclose(R @ R.T, np.eye(3)))
print("round-trip error:", np.abs(R.T @ (R @ np.array([1.0, 2.0, 3.0]))
                                  - np.array([1.0, 2.0, 3.0])).max())


# In[2]:


a = 6378137.0                # semi-major axis, m
e = 0.081819190842622        # eccentricity
lat = np.radians(40.98)
r = a / np.sqrt(1.0 - (e*np.sin(lat))**2)
print("prime vertical radius of curvature at Dunhuang: %.1f m" % r)

