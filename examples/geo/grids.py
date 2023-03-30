"""
source: https://github.com/kumikokotera/GRAND_tools/blob/master/grid_shape/grids.py
"""

from __future__ import absolute_import
import numpy as np
#import astropy.units as u
import logging
import os
import sys
import inspect
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

try:
    from grid_shape import hexy as hx
except:
    import hexy as hx

Z_SITE = 1086 # height of GP300 site in km

def remove_redundant_point(grid_x, grid_y, **kwargs):
    x_pos_flat_fl = grid_x
    y_pos_flat_fl = grid_y
    scal = (x_pos_flat_fl - x_pos_flat_fl.min()) + (x_pos_flat_fl.max() - x_pos_flat_fl.min()) * (y_pos_flat_fl - y_pos_flat_fl.min())
    scal = np.floor(scal)
    unique, index = np.unique(scal, return_index=True)
    x_pos_new = grid_x[index]
    y_pos_new = grid_y[index]
    if "mask" in kwargs:
        return x_pos_new, y_pos_new, kwargs["mask"][index]

    else:
        return x_pos_new, y_pos_new
   
    
def get_hexarray(n_ring, radius, do_mask=False):
    radius_grid = (1 + n_ring *1.5) * 2 / np.sqrt(3) * radius # radius of the circle enclosing the hexgrid
    xcube = hx.get_spiral(np.array((0,0,0)), 0, n_ring)
    xpix = hx.cube_to_pixel(xcube, radius) # centers of the hexagons
    xcorn = hx.get_corners(xpix, radius) # corners of the hexagons
    sh = np.array(xcorn).shape
    xcorn2=xcorn.transpose(0,2,1)
    hexarray = np.array(xcorn2).reshape((sh[0]*sh[2],sh[1]))

    if do_mask: # generate boolean array: true for antennas to keep  
        N_corner = hexarray.shape[0]
        N_pix = xpix.shape[0]

        mask_corner = np.array(np.ones((N_corner, 1)), dtype=bool)
        mask_pix = np.array(np.zeros((N_pix, 1)), dtype=bool)

        hexarray = np.vstack([hexarray, xpix])
        mask = np.vstack([mask_corner, mask_pix])
        return hexarray, radius_grid,  mask
    else:
        return hexarray, radius_grid   


def create_grid_univ(
    GridShape,
    radius,
    angle=0,
    do_offset=False,
    Nrand=None,
    randeff=None,
    DISPLAY=False,
    directory=None,
    do_prune=False, 
    input_n_ring = None
):
    '''
    generate new positions of antennas with universal layout
    write positions in file directory/new_antpos.dat
    should be called outside of database reading loop
    Parameters:
    GridShape: str
        shape of antenna grid
        'rect' = rectangles tiled over rectangular shape
        'hexhex' = hexagons tiled in overall hexagonal shape
        'hexrand' = hexagons tiled in overal hexagonal shape with Nrand randomly displaced antennas
        'trihex' = hexagons + centers (forming 6 triangles) tiled in overall hexagonal shape
    radius: float
        radius of hexagon in m, radius >=2
    angle: float
        angle in degree, rotation angle of grid
        use -theta, when theta is the azimuthal angle of a shower
    do_offset: boolean
        do random offset on antenna positions, keeping (0,0) inside the central cell
    Nrand: int
        for hexrand option: number of randomly displaced antennas
    randeff:
        for hexrand option: antennas are displaced following a normal law
        centered on 0 and of sigma radius/randeff
    directory: str
        path of root directory of shower library
    do_prune: create a mask of the centers of the hexagons in the trihex case
    Output:
    new_pos: numpy arrays
        x, y, z coordinates of antenna in new layout
    offset: list
        x,y coordinates of the grid offset
    mask: bool array
        if do_prune==True and trihex case
    '''

    z_site = Z_SITE # height of GP300 site in km
    try:
        assert radius > 2, "radius must be > 2m"
    except AssertionError:
        sys.exit("radius must be > 2m")

    try: 
        assert GridShape in ["rect", "hexhex", "hexrand", "trihex"]
    except AssertionError:
        sys.exit("Gridshape not known")

    


    if GridShape == 'rect':
        # create rectangular grid
        logging.debug('create_grid: Generating rectangular grid...')

        xNum = 15
        yNum = 15

        grid_x, grid_y = np.mgrid[0:xNum*radius:radius, 0:yNum*radius:radius]

        grid_x = grid_x - (xNum-1)*radius/2
        grid_y = grid_y - (yNum-1)*radius/2

        # flatten grids
        x_pos_new = grid_x.flatten()
        y_pos_new = grid_y.flatten()

    if GridShape == 'hexhex':
        # create a hexagonal grid with overall hexagonal layout
        logging.debug('create_grid:Generating hexagonal grid in hex layout...')
        if input_n_ring:  
            n_ring = input_n_ring
        else:
            n_ring = 5  # number of hex rings corresponding to 216/150 antennas (n_ring=5/4)
        
        hexarray, radius_grid = get_hexarray(n_ring, radius)

        grid_x = hexarray[:,0]
        grid_y = hexarray[:,1]

        # remove redundant points
        x_pos_new, y_pos_new = remove_redundant_point(grid_x, grid_y)

    if GridShape == 'trihex':
        # create a triangular grid with overall hexagonal layout: use a hexagonal grid and add the central point in each cell
        logging.debug('create_grid:Generating triangular grid in overall hexagonal layout...')
        if input_n_ring:  
            n_ring = input_n_ring
        else:
            n_ring = 4 # number of hex rings corresponding to 211 antennas (n_ring=4)
        hexarray, radius_grid,  mask = get_hexarray(n_ring, radius, do_mask=True)

        grid_x = hexarray[:,0]
        grid_y = hexarray[:,1]

        # remove redundant points
        x_pos_new, y_pos_new, mask = remove_redundant_point(grid_x, grid_y, mask=mask)
       
    if GridShape == 'hexrand':
        # create a hexagonal grid with overall hexagonal layout
        logging.debug('create_grid:Generating hexagonal grid in hex layout with random displacements...')

        if input_n_ring:  
            n_ring = input_n_ring
        else:
            n_ring = 4 # number of hex rings corresponding to 216/150 antennas (n_ring=5/4)

        hexarray, radius_grid = get_hexarray(n_ring, radius)

        grid_x = hexarray[:,0]
        grid_y = hexarray[:,1]

        # remove redundant points
        x_pos_new, y_pos_new = remove_redundant_point(grid_x, grid_y)

        # displace Nrand antennas randomly
        Nant = x_pos_new.shape[0]
        indrand = np.random.randint(0, high=Nant, size=Nrand)
        x_pos_new[indrand] += np.random.randn(Nrand) * radius*randeff
        y_pos_new[indrand] += np.random.randn(Nrand) * radius*randeff

  
    # for now set position of z to site altitude
    z_pos_new = x_pos_new*0 + z_site

    # create new position array
    new_pos = np.stack((x_pos_new,y_pos_new,z_pos_new), axis=0)


   # rotate grid of specified angle
    if angle != 0:
        X = new_pos[0,:]
        Y = new_pos[1,:]

        theta = angle / 180 * np.pi

        Xp = X * np.cos(theta) - Y * np.sin(theta)
        Yp = X * np.sin(theta) + Y * np.cos(theta)

        new_pos[0,:] = Xp
        new_pos[1,:] = Yp

    # offset grid of random offset. The (0,0) point lies in the first cell.
    # takes into account the rotation done previously
    if do_offset:

        # offset = get_offset(radius, GridShape)
        if GridShape == "rect":
            x_radius = xNum * radius
            y_radius = yNum * radius
            offset = get_offset_in_grid(GridShape, x_radius, y_radius)
        elif (GridShape == "hexhex") or (GridShape == "trihex") :
            #x_radius = radius
            offset = get_offset_in_grid(GridShape, radius_grid)

        x = offset[0]
        y = offset[1]
        theta = angle / 180 * np.pi
        xp = x * np.cos(theta) - y * np.sin(theta)
        yp = x * np.sin(theta) + y * np.cos(theta)

        new_pos[0,:] -= xp
        new_pos[1,:] -= yp

    else:
        offset = [0,0]
  
    # write new antenna position file
    if(directory!=None):
        logging.debug('create_grid: Writing in file '+ directory +'/new_antpos.dat...')
        FILE = open(directory+ '/new_antpos.dat',"w+" )
        for i in range( 1, len(x_pos_new)+1 ):
            print("%i A%i %1.5e %1.5e %1.5e" % (i,i-1,x_pos_new[i-1],y_pos_new[i-1],z_site), end='\n', file=FILE)
        FILE.close() 

    if DISPLAY:
        fig, axs = plt.subplots(1,1)
        axs.plot(new_pos[0,:], new_pos[1,:], 'r.')

        axs.plot(0,0, 'b.')
        axs.plot(offset[0],offset[1], 'g.')
        axs.axis('equal')
        plt.show()
    
    if do_prune and GridShape=='trihex':
        return new_pos, offset, mask
    else:
        return new_pos, offset


def get_offset(radius, GridShape):
    '''
    Draw random offset in the central cell
    '''

    if GridShape == "rect":
        offset = (np.random.rand(2) - 0.5 ) * radius
    elif (GridShape == "hexhex") or (GridShape == "trihex"):
        offset = (np.random.rand(2) - 0.5 ) * 2*radius
        while not(hx.is_inside_hex(offset, radius)):
            offset = (np.random.rand(2) - 0.5 ) * 2*radius
    else:
        print("offset not yet implemented for this GridShape")
        offset=[0,0]
    return offset


def get_offset_in_grid(GridShape, x_radius=None, y_radius=None):
    '''
    Draw random offset in the original grid
    '''

    if GridShape == "rect":
        x_offset = (np.random.rand(1) - 0.5 ) * x_radius
        y_offset = (np.random.rand(1) - 0.5 ) * y_radius
    elif (GridShape == "hexhex") or (GridShape == "trihex"):
        x_offset, y_offset = (np.random.rand(2) - 0.5 ) * 2 * x_radius
        while not(hx.is_inside_hex_flattop([x_offset, y_offset], x_radius)):
            x_offset, y_offset = (np.random.rand(2) - 0.5 ) * 2 *x_radius

    else:
        print("offset not yet implemented for this GridShape")
        offset=[0,0]
    return [x_offset, y_offset]