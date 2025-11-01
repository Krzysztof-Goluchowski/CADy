# Python translation of the MATLAB bitmap_h function
# Keeps the algorithm, data structures and indexing behaviour similar to the original
# Requirements: Pillow, numpy, matplotlib
# Usage example:
#   from bitmap_h import bitmap_h
#   bitmap_h("mp.JPG", 10, 10, 0.1, 3, False)

import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def bitmap_h(filename, elementsx, elementsy, maxerror, max_refinement_level, color_edges_black, channel='all'):
    """
    h-adaptive bitmap projection translated from MATLAB.
    Parameters mirror the original function.
    Note: internally coordinates are stored 1-based (to stay close to MATLAB logic).
    """

    # read image from file
    img = Image.open(filename).convert('RGB')
    XX = np.array(img)  # shape (rows, cols, 3)

    RR = np.zeros_like(XX[:, :, 0])
    GG = np.zeros_like(XX[:, :, 0])
    BB = np.zeros_like(XX[:, :, 0])

    if channel == 'R' :
        RR = XX[:, :, 0].astype(np.int32)
    elif channel == 'G' :
        GG = XX[:, :, 1].astype(np.int32)
    elif channel == 'B' :
        BB = XX[:, :, 2].astype(np.int32)
    elif channel == 'all' :
        RR = XX[:, :, 0].astype(np.int32)
        GG = XX[:, :, 1].astype(np.int32)
        BB = XX[:, :, 2].astype(np.int32)
    else :
        print(f"Unknown channel: '{channel}'")
        print(f"Available values: 'R', 'G', 'B', 'all'")
        return None

    # read size of image (mx rows, my cols) -- keep MATLAB names
    ix = XX.shape[0]
    iy = XX.shape[1]

    # global counters and containers
    total_vertexes = 0
    total_elements = 0

    # elements and vertexes lists (1-based simulation: we'll append and index by -1)
    elements = []  # each element is a dict with fields in the MATLAB code
    vertexes = []  # each vertex is a dict

    # helper to convert our 1-based vertex coordinates to numpy indices
    def pix(x, y):
        # x,y are 1-based coordinates (row, col)
        # clamp to image bounds and return integer indices for numpy (0-based)
        xr = min(max(int(x) - 1, 0), ix - 1)
        yr = min(max(int(y) - 1, 0), iy - 1)
        return xr, yr

    # create vertex (non-hanging)
    def create_vertex(x, y):
        nonlocal total_vertexes
        total_vertexes += 1
        xr, yr = pix(x, y)
        vert = {
            'x': int(x),
            'y': int(y),
            'r': int(RR[xr, yr]),
            'g': int(GG[xr, yr]),
            'b': int(BB[xr, yr]),
            'index': total_vertexes,
            'real': True
        }
        vertexes.append(vert)
        return total_vertexes

    # create vertex (hanging) with explicit rgb
    def create_vertex_rgb(x, y, r, g, b):
        nonlocal total_vertexes
        total_vertexes += 1
        vert = {
            'x': int(x),
            'y': int(y),
            'r': int(r),
            'g': int(g),
            'b': int(b),
            'index': total_vertexes,
            'real': False
        }
        vertexes.append(vert)
        return total_vertexes

    # update vertex when hanging node becomes real
    def vert_update(index):
        # index is 1-based
        v = vertexes[index - 1]
        xr, yr = pix(v['x'], v['y'])
        v['r'] = int(RR[xr, yr])
        v['g'] = int(GG[xr, yr])
        v['b'] = int(BB[xr, yr])
        v['real'] = True
        vertexes[index - 1] = v

    # create element (returns element dict and increments total_elements)
    def create_element(v1, v2, v3, v4):
        nonlocal total_elements
        total_elements += 1
        element = {
            'dl': v1, 'ul': v2, 'dr': v3, 'ur': v4,
            'active': True,
            'elu': 0, 'eld': 0, 'edl': 0, 'edr': 0,
            'eul': 0, 'eur': 0, 'eru': 0, 'erd': 0,
            'index': total_elements
        }
        elements.append(element)
        return element

    # initialize mesh
    def init_mesh():
        # mapping in original code: ix rows, iy cols
        elem_width = math.floor(ix / elementsx)
        elem_hight = math.floor(iy / elementsy)

        x = 0
        y = 0

        # create all vertexes (note: original code used 1-based indexing)
        for i in range(0, elementsy):
            for j in range(0, elementsx):
                create_vertex(x + j * elem_width + 1, y + 1)
            create_vertex(ix, y + 1)
            y = y + elem_hight
        for j in range(0, elementsx):
            create_vertex(x + j * elem_width + 1, iy)
        create_vertex(ix, iy)

        # create all elements
        # careful: original MATLAB indexing for v1..v4 used a formula; translate faithfully
        # in MATLAB: v1 = (i-1)*elementsy+j + i-1; v3 = v1+1; v2 = i*elementsy+j+1 + i-1; v4 = v2+1;
        for i in range(1, elementsy + 1):
            for j in range(1, elementsx + 1):
                v1 = (i - 1) * elementsy + j + i - 1
                v3 = v1 + 1
                v2 = i * elementsy + j + 1 + i - 1
                v4 = v2 + 1
                el = create_element(v1, v2, v3, v4)
                idx = el['index']
                # set neighbours
                if j != 1:
                    el['elu'] = idx - 1
                if j != elementsx:
                    el['eru'] = idx + 1
                if i != 1:
                    el['edl'] = idx - elementsx
                if i != elementsy:
                    el['eul'] = idx + elementsx
                # update list entry (it is already appended)
                elements[idx - 1] = el

    # interpolate rgb for a hanging node between vertices v1 and v2 of given element
    def interpolate_rgb(v1, v2, element_index):
        nonlocal total_vertexes
        elem = elements[element_index - 1]
        width = vertexes[elem['dr'] - 1]['x'] - vertexes[elem['dl'] - 1]['x']
        hight = vertexes[elem['ul'] - 1]['y'] - vertexes[elem['dl'] - 1]['y']
        vert1 = vertexes[v1 - 1]
        vert2 = vertexes[v2 - 1]

        vert3x = math.floor((vert1['x'] + vert2['x']) / 2)
        vert3y = math.floor((vert1['y'] + vert2['y']) / 2)

        xx = vert3x - vertexes[elem['dl'] - 1]['x']
        yy = vert3y - vertexes[elem['dl'] - 1]['y']
        r, g, b = inpoint(xx, yy, width, hight, elem)

        total_vertexes += 1
        vert3 = {
            'x': vert3x,
            'y': vert3y,
            'r': int(r),
            'g': int(g),
            'b': int(b),
            'index': total_vertexes,
            'real': False
        }
        vertexes.append(vert3)
        return total_vertexes

    # compute r,g,b for each point inside element and write to RR,GG,BB
    def interpolate_elem(element_index, color_edges_black):
        elem = elements[element_index - 1]
        width = vertexes[elem['dr'] - 1]['x'] - vertexes[elem['dl'] - 1]['x']
        hight = vertexes[elem['ul'] - 1]['y'] - vertexes[elem['dl'] - 1]['y']
        width = abs(width)
        hight = abs(hight)
        dlx = vertexes[elem['dl'] - 1]['x']
        dly = vertexes[elem['dl'] - 1]['y']

        for xx in range(0, width + 1):
            for yy in range(0, hight + 1):
                r, g, b = inpoint(xx, yy, width, hight, elem)
                xr, yr = pix(dlx + xx, dly + yy)
                RR[xr, yr] = int(r)
                GG[xr, yr] = int(g)
                BB[xr, yr] = int(b)

        if color_edges_black:
            for xx in range(0, width + 1):
                xr, yr = pix(dlx + xx, dly)
                RR[xr, yr] = 0
                GG[xr, yr] = 0
                BB[xr, yr] = 0
                xr, yr = pix(dlx + xx, dly + hight)
                RR[xr, yr] = 0
                GG[xr, yr] = 0
                BB[xr, yr] = 0
            for yy in range(0, hight + 1):
                xr, yr = pix(dlx, dly + yy)
                RR[xr, yr] = 0
                GG[xr, yr] = 0
                BB[xr, yr] = 0
                xr, yr = pix(dlx + width, dly + yy)
                RR[xr, yr] = 0
                GG[xr, yr] = 0
                BB[xr, yr] = 0

    # basis interpolation inside element
    def inpoint(xx, yy, width, hight, elem):
        def fi1(xx, yy):
            x = xx / width if width != 0 else 0
            y = yy / hight if hight != 0 else 0
            return (1 - x) * (1 - y)

        def fi2(xx, yy):
            x = xx / width if width != 0 else 0
            y = yy / hight if hight != 0 else 0
            return (1 - x) * y

        def fi3(xx, yy):
            x = xx / width if width != 0 else 0
            y = yy / hight if hight != 0 else 0
            return x * (1 - y)

        def fi4(xx, yy):
            x = xx / width if width != 0 else 0
            y = yy / hight if hight != 0 else 0
            return x * y

        f1 = fi1(xx, yy)
        f2 = fi2(xx, yy)
        f3 = fi3(xx, yy)
        f4 = fi4(xx, yy)

        r = (vertexes[elem['dl'] - 1]['r'] * f1 + vertexes[elem['ul'] - 1]['r'] * f2
             + vertexes[elem['dr'] - 1]['r'] * f3 + vertexes[elem['ur'] - 1]['r'] * f4)
        g = (vertexes[elem['dl'] - 1]['g'] * f1 + vertexes[elem['ul'] - 1]['g'] * f2
             + vertexes[elem['dr'] - 1]['g'] * f3 + vertexes[elem['ur'] - 1]['g'] * f4)
        b = (vertexes[elem['dl'] - 1]['b'] * f1 + vertexes[elem['ul'] - 1]['b'] * f2
             + vertexes[elem['dr'] - 1]['b'] * f3 + vertexes[elem['ur'] - 1]['b'] * f4)

        return math.floor(r), math.floor(g), math.floor(b)

    # check neighbours - break if they are bigger
    def break_neighbours(index):
        element = elements[index - 1]

        def check_left():
            if element['elu'] == 0:
                return
            if element['eld'] != 0:
                return
            left = elements[element['elu'] - 1]
            if left['erd'] != 0:
                break_element(element['elu'])

        def check_right():
            if element['eru'] == 0:
                return
            if element['erd'] != 0:
                return
            right = elements[element['eru'] - 1]
            if right['eld'] != 0:
                break_element(element['eru'])

        def check_up():
            if element['eul'] == 0:
                return
            if element['eur'] != 0:
                return
            up = elements[element['eul'] - 1]
            if up['edr'] != 0:
                break_element(element['eul'])

        def check_down():
            if element['edl'] == 0:
                return
            if element['edr'] != 0:
                return
            down = elements[element['edl'] - 1]
            if down['eur'] != 0:
                break_element(element['edl'])

        check_left()
        check_right()
        check_up()
        check_down()

    # breaking element (subdivide into 4)
    def break_element(index):
        element = elements[index - 1]
        if not element['active']:
            print('error!!!')
        # ensure neighbours that must be split are split as well
        break_neighbours(index)
        element = elements[index - 1]

        v1 = element['dl']
        v2 = element['ul']
        v3 = element['dr']
        v4 = element['ur']

        v5 = 0; v6 = 0; v7 = 0; v8 = 0; v9 = 0

        # left side
        if element['eld'] != 0:
            eld = elements[element['eld'] - 1]
            v6 = eld['ur']
            vert_update(v6)
        else:
            v6 = interpolate_rgb(v1, v2, index)
        if element['elu'] == 0:
            vert_update(v6)

        # right side
        if element['erd'] != 0:
            erd = elements[element['erd'] - 1]
            v8 = erd['ul']
            vert_update(v8)
        else:
            v8 = interpolate_rgb(v3, v4, index)
        if element['eru'] == 0:
            vert_update(v8)

        # up
        if element['eur'] != 0:
            eur = elements[element['eur'] - 1]
            v9 = eur['dl']
            vert_update(v9)
        else:
            v9 = interpolate_rgb(v2, v4, index)
        if element['eul'] == 0:
            vert_update(v9)

        # down
        if element['edr'] != 0:
            edr = elements[element['edr'] - 1]
            v5 = edr['ul']
            vert_update(v5)
        else:
            v5 = interpolate_rgb(v1, v3, index)
        if element['edl'] == 0:
            vert_update(v5)

        x = vertexes[v5 - 1]['x']
        y = vertexes[v6 - 1]['y']
        v7 = create_vertex(x, y)

        # deactivate parent
        element['active'] = False
        elements[element['index'] - 1] = element

        e1 = create_element(v1, v6, v5, v7)
        e2 = create_element(v6, v2, v7, v9)
        e3 = create_element(v5, v7, v3, v8)
        e4 = create_element(v7, v9, v8, v4)

        # set neighbours among new elements
        e1['eru'] = e3['index']
        e1['eul'] = e2['index']
        e2['edl'] = e1['index']
        e2['eru'] = e4['index']
        e3['elu'] = e1['index']
        e3['eul'] = e4['index']
        e4['elu'] = e2['index']
        e4['edl'] = e3['index']

        # connect new with old neighbors (faithful translation)
        e1['edl'] = element['edl']
        if element['edl'] != 0:
            edl = elements[element['edl'] - 1]
            edl['eul'] = e1['index']
            elements[edl['index'] - 1] = edl

        if element['edr'] != 0:
            e3['edl'] = element['edr']
            edr = elements[element['edr'] - 1]
            edr['eul'] = e3['index']
            elements[edr['index'] - 1] = edr
        else:
            e3['edl'] = element['edl']
            if element['edl'] != 0:
                edl = elements[element['edl'] - 1]
                edl['eur'] = e3['index']
                elements[edl['index'] - 1] = edl

        e2['elu'] = element['elu']
        if element['elu'] != 0:
            elu = elements[element['elu'] - 1]
            elu['eru'] = e2['index']
            elements[elu['index'] - 1] = elu

        if element['eld'] != 0:
            e1['elu'] = element['eld']
            eld = elements[element['eld'] - 1]
            eld['eru'] = e1['index']
            elements[eld['index'] - 1] = eld
        else:
            e1['elu'] = element['elu']
            if element['elu'] != 0:
                elu = elements[element['elu'] - 1]
                elu['erd'] = e1['index']
                elements[elu['index'] - 1] = elu

        e2['eul'] = element['eul']
        if element['eul'] != 0:
            eul = elements[element['eul'] - 1]
            eul['edl'] = e2['index']
            elements[eul['index'] - 1] = eul
        if element['eur'] != 0:
            e4['eul'] = element['eur']
            eur = elements[element['eur'] - 1]
            eur['edl'] = e4['index']
            elements[eur['index'] - 1] = eur
        else:
            e4['eul'] = element['eul']
            if element['eul'] != 0:
                eul = elements[element['eul'] - 1]
                eul['edr'] = e4['index']
                elements[eul['index'] - 1] = eul

        e4['eru'] = element['eru']
        if element['eru'] != 0:
            eru = elements[element['eru'] - 1]
            eru['elu'] = e4['index']
            elements[eru['index'] - 1] = eru

        if element['erd'] != 0:
            e3['eru'] = element['erd']
            erd = elements[element['erd'] - 1]
            erd['elu'] = e3['index']
            elements[erd['index'] - 1] = erd
        else:
            e3['eru'] = element['eru']
            if element['eru'] != 0:
                eru = elements[element['eru'] - 1]
                eru['eld'] = e3['index']
                elements[eru['index'] - 1] = eru

        # store updated new elements
        elements[e4['index'] - 1] = e4
        elements[e3['index'] - 1] = e3
        elements[e2['index'] - 1] = e2
        elements[e1['index'] - 1] = e1

    # estimate relative interpolation error on element
    def estimate_error(index):
        element = elements[index - 1]
        dl = element['dl']
        ul = element['ul']
        dr = element['dr']
        ur = element['ur']

        xl = vertexes[dl - 1]['x']
        yd = vertexes[dl - 1]['y']
        xr = vertexes[ur - 1]['x']
        yu = vertexes[ur - 1]['y']

        elementWidth = xr - xl
        elementHeigth = yu - yd

        # Gaussian quadrature points (2x2)
        x1 = elementWidth / 2.0 - elementWidth / (math.sqrt(3.0) * 2.0)
        x2 = elementWidth / 2.0 + elementWidth / (math.sqrt(3.0) * 2.0)
        y1 = elementHeigth / 2.0 - elementHeigth / (math.sqrt(3.0) * 2.0)
        y2 = elementHeigth / 2.0 + elementHeigth / (math.sqrt(3.0) * 2.0)

        x1 = math.floor(x1)
        x2 = math.floor(x2)
        y1 = math.floor(y1)
        y2 = math.floor(y2)

        r1, g1, b1 = inpoint(x1, y1, elementWidth, elementHeigth, element)
        r2, g2, b2 = inpoint(x1, y2, elementWidth, elementHeigth, element)
        r3, g3, b3 = inpoint(x2, y1, elementWidth, elementHeigth, element)
        r4, g4, b4 = inpoint(x2, y2, elementWidth, elementHeigth, element)

        # sample actual image at those points
        xr1, yr1 = pix(x1 + xl, y1 + yd)
        xr2, yr2 = pix(x1 + xl, y2 + yd)
        xr3, yr3 = pix(x2 + xl, y1 + yd)
        xr4, yr4 = pix(x2 + xl, y2 + yd)

        r1 = r1 - int(RR[xr1, yr1])
        g1 = g1 - int(GG[xr1, yr1])
        b1 = b1 - int(BB[xr1, yr1])

        r2 = r2 - int(RR[xr2, yr2])
        g2 = g2 - int(GG[xr2, yr2])
        b2 = b2 - int(BB[xr2, yr2])

        r3 = r3 - int(RR[xr3, yr3])
        g3 = g3 - int(GG[xr3, yr3])
        b3 = b3 - int(BB[xr3, yr3])

        r4 = r4 - int(RR[xr4, yr4])
        g4 = g4 - int(GG[xr4, yr4])
        b4 = b4 - int(BB[xr4, yr4])

        error_r = r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4
        error_g = g1 * g1 + g2 * g2 + g3 * g3 + g4 * g4
        error_b = b1 * b1 + b2 * b2 + b3 * b3 + b4 * b4

        error_r = float(error_r)
        error_g = float(error_g)
        error_b = float(error_b)

        error_r = math.sqrt(error_r) * 100.0 / (255.0 * 2.0)
        error_g = math.sqrt(error_g) * 100.0 / (255.0 * 2.0)
        error_b = math.sqrt(error_b) * 100.0 / (255.0 * 2.0)

        return error_r, error_g, error_b

    # ------ main algorithm ------
    init_mesh()

    redo_error_test = True
    refinemenet_level = 0

    while redo_error_test and (refinemenet_level < max_refinement_level):
        redo_error_test = False
        # loop through elements
        # must snapshot total_elements because elements may grow while iterating
        cur_total = total_elements
        for i in range(1, cur_total + 1):
            if elements[i - 1]['active']:
                rr, gg, bb = estimate_error(i)
                if (rr >= maxerror) or (gg >= maxerror) or (bb >= maxerror):
                    redo_error_test = True
                    break_element(i)
        refinemenet_level += 1

    # interpolate all active elements - recreate bitmap r,g,b
    cur_total = total_elements
    for i in range(1, cur_total + 1):
        if elements[i - 1]['active']:
            interpolate_elem(i, color_edges_black)

    # recreate RGB image
    RGB = np.stack([RR, GG, BB], axis=2).astype(np.uint8)

    # display image (similar to imshow in MATLAB)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(RGB)
    plt.show()

    print('total_elements =', total_elements)


if __name__ == "__main__" :
    bitmap_h("Comet.jpg",10,10,1,8,False)
    bitmap_h("Comet.jpg",10,10,1,8,False, 'R')
    bitmap_h("Comet.jpg",10,10,1,8,False, 'G')
    bitmap_h("Comet.jpg",10,10,1,8,False, 'B')
