''' In-house functions to produce translational, radial, and rotational dot
motion image sequences.

[DEPENDENCIES]
+ numpy==1.15.4
+ scipy

'''
#  Helper libraries
import numpy as np
import scipy.ndimage as ndimage


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def init_dots(dot_size, imRad, n_dots):
    dot_pol = np.random.choice([-1, 1], n_dots)
    py = np.random.uniform(-imRad+dot_size, imRad-dot_size, n_dots)
    px = np.random.uniform(-imRad+dot_size, imRad-dot_size, n_dots)
    return(py, px, dot_pol)


def move_dots(dot_size, imRad, border, py, px, dx, dy, dz, dr):
    rho, phi = cart2pol(py+dy, px+dx)
    DZ = rho/imRad*dz*2
    if np.size(np.array(dz)) == 1:
        chk = dz
    else:
        chk = dz[-1]
    if chk < 0:
        repos = np.where(((rho+DZ) < dot_size/2)
                         | ((rho+dz) > (imRad-dot_size/2)))[0]
    else:
        repos = np.where(((rho+DZ) > (imRad-dot_size/2)))[0]
    if chk > 0:
        py[repos] = np.random.uniform(-imRad+dot_size/2+border+np.abs(dy).max()+np.abs(
            dz).max(), imRad-dot_size/2-border-np.abs(dy).max()-np.abs(dz).max(), len(repos))
        px[repos] = np.random.uniform(-imRad+dot_size/2+border+np.abs(dx).max()+np.abs(
            dz).max(), imRad-dot_size/2-border-np.abs(dx).max()-np.abs(dz).max(), len(repos))
    else:
        py[repos] = np.random.uniform(-imRad+dot_size/2+border+np.abs(
            dy).max(), imRad-dot_size/2-border-np.abs(dy).max(), len(repos))
        px[repos] = np.random.uniform(-imRad+dot_size/2+border+np.abs(
            dx).max(), imRad-dot_size/2-border-np.abs(dx).max(), len(repos))

    rho, phi = cart2pol(py+dy, px+dx)
    DZ = rho/imRad*dz*2
    py, px = pol2cart(rho+DZ, phi-dr)
    return(py, px)


def draw_dots(py, px, cry, crx, imRad, n_dots, dot_mat, imSize, border, scale, dot_pol):
    dstRect = np.array([py-cry, px-crx, py+cry, px+crx]+imRad).astype(int)
    bg = np.zeros([imRad*2, imRad*2])
    for d_idx in range(n_dots):
        bg[dstRect[0, d_idx]:dstRect[2, d_idx], dstRect[1, d_idx]:dstRect[3, d_idx]] += dot_mat*dot_pol[d_idx]
    bg[bg > 1] = 1
    img = ndimage.zoom(
        bg[border:border+imSize, border:border+imSize], zoom=1/scale, order=1)
    return(img)
