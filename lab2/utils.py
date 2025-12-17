import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import random
import plotly.graph_objects as go

def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def get_transformed_pixels_coords(I, H, shift=None):
    ys, xs = np.indices(I.shape[:2]).astype("float64")
    if shift is not None:
        ys += shift[1]
        xs += shift[0]
    ones = np.ones(I.shape[:2])
    coords = np.stack((xs, ys, ones), axis=2)
    coords_H = (H @ coords.reshape(-1, 3).T).T
    coords_H /= coords_H[:, 2, np.newaxis]
    cart_H = coords_H[:, :2]
    
    return cart_H.reshape((*I.shape[:2], 2))

def apply_H_pix(pix, H):
    x, y = pix  # x=columna, y=fila
    v = np.array([x, y, 1.0], dtype=float)
    xp, yp, wp = H @ v
    return np.array([xp / wp, yp / wp], dtype=float)

def apply_H_fixed_image_size(I, H, corners):
    I = np.asarray(I)
    H = np.asarray(H, dtype=float)
    invH = np.linalg.inv(H)

    # Soporta gris o RGB/RGBA
    if I.ndim == 2:
        I = I[..., None]
    h, w, channels = I.shape

    corners = np.asarray(corners, dtype=float)

    # corners sera: (min_x, max_x, min_y, max_y)
    min_x, max_x, min_y, max_y = corners
    min_x, min_y = np.floor(min_x), np.floor(min_y)
    max_x, max_y = np.ceil(max_x), np.ceil(max_y)
    

    out_w = int(max_x - min_x + 1)
    out_h = int(max_y - min_y + 1)

    out = np.zeros((out_h, out_w, channels), dtype=I.dtype)
    I_float = I.astype(float)

    for j_out in range(out_h):      # y (fila)
        y_out = j_out + min_y
        for i_out in range(out_w):  # x (columna)
            x_out = i_out + min_x

            # backwarp: destino -> origen
            x_in, y_in = apply_H_pix((x_out, y_out), invH)

            coords = [[y_in], [x_in]]  # (fila, columna)
            for c in range(channels):
                out[j_out, i_out, c] = map_coordinates(
                    I_float[:, :, c],
                    coords,
                    order=1,
                    mode="constant",
                    cval=0.0
                )[0]

    # Si era gris, devolver gris
    if out.shape[2] == 1:
        out = out[:, :, 0]
    return out


def Normalise_last_coord(x):
    xn = x  / x[2,:]
    
    return xn

def _normalize_points_2d(points):
    """
    Normalización de Hartley para puntos 2D homogéneos.
    points: 3xN (homogéneos)
    Devuelve: points_norm (3xN), T (3x3) tal que points_norm = T @ points
    """
    points = np.asarray(points, dtype=float)
    assert points.shape[0] == 3

    # Pasar a inhomogéneos
    x = points[0, :] / points[2, :]
    y = points[1, :] / points[2, :]

    cx = np.mean(x)
    cy = np.mean(y)

    x_c = x - cx
    y_c = y - cy

    mean_dist = np.mean(np.sqrt(x_c**2 + y_c**2))
    if mean_dist < 1e-12:
        s = 1.0
    else:
        s = np.sqrt(2) / mean_dist

    T = np.array([
        [s, 0, -s * cx],
        [0, s, -s * cy],
        [0, 0, 1]
    ], dtype=float)

    points_norm = T @ points
    return points_norm, T


def DLT_homography(points1, points2):
    """
    Calcula H tal que points2 ~ H @ points1 usando DLT (con normalización).
    points1, points2: 3xN
    """
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)

    if points1.shape[0] != 3 or points2.shape[0] != 3:
        raise ValueError("points1 y points2 deben ser 3xN (homogéneos)")
    if points1.shape[1] < 4:
        raise ValueError("Se necesitan al menos 4 correspondencias para homografía")

    # Normalizar
    p1n, T1 = _normalize_points_2d(points1)
    p2n, T2 = _normalize_points_2d(points2)

    N = p1n.shape[1]
    A = np.zeros((2 * N, 9), dtype=float)

    x1 = p1n[0, :] / p1n[2, :]
    y1 = p1n[1, :] / p1n[2, :]
    x2 = p2n[0, :] / p2n[2, :]
    y2 = p2n[1, :] / p2n[2, :]

    for i in range(N):
        X, Y = x1[i], y1[i]
        u, v = x2[i], y2[i]

        A[2*i,   :] = [-X, -Y, -1,  0,  0,  0,  u*X, u*Y, u]
        A[2*i+1, :] = [ 0,  0,  0, -X, -Y, -1,  v*X, v*Y, v]

    # Resolver Ah = 0 con SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]                 # vector asociado al menor singular
    Hn = h.reshape(3, 3)

    # Desnormalizar: points2 ~ inv(T2) @ Hn @ T1 @ points1
    H = np.linalg.inv(T2) @ Hn @ T1

    # Normalizar escala (opcional pero práctico)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    else:
        H = H / (np.linalg.norm(H) + 1e-12)

    return H


def Inliers(H, points1, points2, th):
    """
    Devuelve los índices de correspondencias inlier según un umbral th (en píxeles).
    Usa error simétrico:
      d = ||p2 - H p1|| + ||p1 - H^-1 p2||
    points1, points2: 3xN
    """
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)
    H = np.asarray(H, dtype=float)

    # Check that H is invertible (ya lo tenías)
    if abs(math.log(np.linalg.cond(H))) > 15:
        return np.array([], dtype=int)

    invH = np.linalg.inv(H)

    # Proyección 1 -> 2
    p2_est = H @ points1
    p2_est = p2_est[:2, :] / p2_est[2:3, :]

    p2_true = points2[:2, :] / points2[2:3, :]

    # Proyección 2 -> 1
    p1_est = invH @ points2
    p1_est = p1_est[:2, :] / p1_est[2:3, :]

    p1_true = points1[:2, :] / points1[2:3, :]

    # Error simétrico
    d12 = np.sqrt(np.sum((p2_est - p2_true) ** 2, axis=0))
    d21 = np.sqrt(np.sum((p1_est - p1_true) ** 2, axis=0))
    d = d12 + d21

    idx = np.where(d < th)[0].astype(int)
    return idx


def Ransac_DLT_homography(points1, points2, th, max_it):
    
    Ncoords, Npts = points1.shape
    
    it = 0
    best_inliers = np.empty(1)
    
    while it < max_it:
        indices = random.sample(range(1, Npts), 4)
        H = DLT_homography(points1[:,indices], points2[:,indices])
        inliers = Inliers(H, points1, points2, th)
        
        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
        
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 -  fracinliers**4
        eps = np.finfo(float).eps
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = math.log(1-p)/math.log(pNoOutliers)
        
        it += 1
    
    # compute H from all the inliers
    H = DLT_homography(points1[:,best_inliers], points2[:,best_inliers])
    inliers = best_inliers
    
    return H, inliers



def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o

def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:,:3]) @ np.array([x[0], x[1], 1])
    return v

def plot_camera(P, w, h, fig, legend):
    
    o = optical_center(P)
    scale = 200
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale
    
    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0]+p2[0])/2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1]+p2[1])/2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2]+p2[2])/2])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return

def plot_image_origin(w, h, fig, legend):
    p1 = np.array([0, 0, 0])
    p2 = np.array([w, 0, 0])
    p3 = np.array([w, h, 0])
    p4 = np.array([0, h, 0])
    
    x = np.array([p1[0], p2[0], p3[0], p4[0], p1[0]])
    y = np.array([p1[1], p2[1], p3[1], p4[1], p1[1]])
    z = np.array([p1[2], p2[2], p3[2], p4[2], p1[2]])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return
