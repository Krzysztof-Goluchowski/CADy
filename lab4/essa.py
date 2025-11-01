import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

def bitmap_h(filename, elementsx, elementsy, maxerror, max_refinement_level, color_edges_black):
    # read image
    XX = np.array(Image.open(filename).convert("RGB"))
    RR = XX[:, :, 0].astype(float)
    GG = XX[:, :, 1].astype(float)
    BB = XX[:, :, 2].astype(float)

    ix, iy = XX.shape[0], XX.shape[1]
    total_vertexes = 0
    total_elements = 0

    elements = []
    vertexes = []

    # ============================ Utility / Structures ============================

    def create_vertex(x, y):
        nonlocal total_vertexes
        vert = {
            'x': int(x),
            'y': int(y),
            'r': RR[int(x)-2, int(y)-2],
            'g': GG[int(x)-2, int(y)-2],
            'b': BB[int(x)-2, int(y)-2],
            'index': total_vertexes + 1,
            'real': True
        }
        vertexes.append(vert)
        total_vertexes += 1
        return total_vertexes

    def create_vertex_rgb(x, y, r, g, b):
        nonlocal total_vertexes
        vert = {
            'x': int(x),
            'y': int(y),
            'r': r,
            'g': g,
            'b': b,
            'index': total_vertexes + 1,
            'real': False
        }
        vertexes.append(vert)
        total_vertexes += 1
        return total_vertexes

    def vert_update(index):
        vert = vertexes[index - 1]
        vert['r'] = RR[int(vert['x']) - 2, int(vert['y']) - 2]
        vert['g'] = GG[int(vert['x']) - 2, int(vert['y']) - 2]
        vert['b'] = BB[int(vert['x']) - 2, int(vert['y']) - 2]
        vert['real'] = True
        vertexes[index - 1] = vert

    def create_element(v1, v2, v3, v4):
        nonlocal total_elements
        elem = {
            'dl': v1, 'ul': v2, 'dr': v3, 'ur': v4,
            'active': True,
            'elu': 0, 'eld': 0, 'edl': 0, 'edr': 0,
            'eul': 0, 'eur': 0, 'eru': 0, 'erd': 0,
            'index': total_elements + 1
        }
        elements.append(elem)
        total_elements += 1
        return elem

    # ============================ Mesh initialization ============================

    def init_mesh():
        nonlocal total_elements, total_vertexes
        elem_width = math.floor(ix / elementsx)
        elem_height = math.floor(iy / elementsy)

        # create vertexes grid
        for i in range(elementsy + 1):
            for j in range(elementsx + 1):
                create_vertex(1 + j * elem_width, 1 + i * elem_height)

        # create elements
        for i in range(elementsy):
            for j in range(elementsx):
                v1 = i * (elementsx + 1) + j + 1
                v2 = (i + 1) * (elementsx + 1) + j + 1
                v3 = i * (elementsx + 1) + j + 2
                v4 = (i + 1) * (elementsx + 1) + j + 2
                elem = create_element(v1, v2, v3, v4)
                idx = elem['index']
                if j != 0:
                    elem['elu'] = idx - 1
                if j != elementsx - 1:
                    elem['eru'] = idx + 1
                if i != 0:
                    elem['edl'] = idx - elementsx
                if i != elementsy - 1:
                    elem['eul'] = idx + elementsx
                elements[idx - 1] = elem

    # ============================ Interpolation Functions ============================

    def inpoint(xx, yy, width, height, elem):
        def fi1(x, y): return (1 - x/width) * (1 - y/height)
        def fi2(x, y): return (1 - x/width) * (y/height)
        def fi3(x, y): return (x/width) * (1 - y/height)
        def fi4(x, y): return (x/width) * (y/height)

        f1, f2, f3, f4 = fi1(xx, yy), fi2(xx, yy), fi3(xx, yy), fi4(xx, yy)
        vdl = vertexes[elem['dl'] - 1]
        vul = vertexes[elem['ul'] - 1]
        vdr = vertexes[elem['dr'] - 1]
        vur = vertexes[elem['ur'] - 1]

        r = math.floor(vdl['r'] * f1 + vul['r'] * f2 + vdr['r'] * f3 + vur['r'] * f4)
        g = math.floor(vdl['g'] * f1 + vul['g'] * f2 + vdr['g'] * f3 + vur['g'] * f4)
        b = math.floor(vdl['b'] * f1 + vul['b'] * f2 + vdr['b'] * f3 + vur['b'] * f4)
        return r, g, b

    def interpolate_elem(element, color_edges_black):
        elem = element
        vdl = vertexes[elem['dl'] - 1]
        vdr = vertexes[elem['dr'] - 1]
        vul = vertexes[elem['ul'] - 1]
        width = abs(vdr['x'] - vdl['x'])
        height = abs(vul['y'] - vdl['y'])
        dlx = vdl['x']
        dly = vdl['y']

        for xx in range(width):
            for yy in range(height):
                r, g, b = inpoint(xx, yy, width, height, elem)
                RR[int(dlx) - 1 + xx, int(dly) - 1 + yy] = r
                GG[int(dlx) - 1 + xx, int(dly) - 1 + yy] = g
                BB[int(dlx) - 1 + xx, int(dly) - 1 + yy] = b

        if color_edges_black:
            for xx in range(width):
                RR[int(dlx) - 1 + xx, int(dly) - 1] = 0
                RR[int(dlx) - 1 + xx, int(dly) - 1 + height] = 0
                GG[int(dlx) - 1 + xx, int(dly) - 1] = 0
                GG[int(dlx) - 1 + xx, int(dly) - 1 + height] = 0
                BB[int(dlx) - 1 + xx, int(dly) - 1] = 0
                BB[int(dlx) - 1 + xx, int(dly) - 1 + height] = 0
            for yy in range(height):
                RR[int(dlx) - 1, int(dly) - 1 + yy] = 0
                RR[int(dlx) - 1 + width, int(dly) - 1 + yy] = 0
                GG[int(dlx) - 1, int(dly) - 1 + yy] = 0
                GG[int(dlx) - 1 + width, int(dly) - 1 + yy] = 0
                BB[int(dlx) - 1, int(dly) - 1 + yy] = 0
                BB[int(dlx) - 1 + width, int(dly) - 1 + yy] = 0

    # ============================ Error Estimation ============================

    def estimate_error(index):
        elem = elements[index - 1]
        vdl, vul, vdr, vur = (vertexes[elem[k]-1] for k in ['dl','ul','dr','ur'])
        xl, yd = vdl['x'], vdl['y']
        xr, yu = vur['x'], vur['y']
        w = xr - xl
        h = yu - yd
        x1 = w/2 - w/(math.sqrt(3)*2)
        x2 = w/2 + w/(math.sqrt(3)*2)
        y1 = h/2 - h/(math.sqrt(3)*2)
        y2 = h/2 + h/(math.sqrt(3)*2)
        pts = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
        err_r = err_g = err_b = 0
        for (x,y) in pts:
            r,g,b = inpoint(x,y,w,h,elem)
            rx,ry = int(xl+x)-1,int(yd+y)-1
            r -= RR[rx,ry]; g -= GG[rx,ry]; b -= BB[rx,ry]
            err_r += r**2; err_g += g**2; err_b += b**2
        err_r = math.sqrt(err_r)*100/(255*2)
        err_g = math.sqrt(err_g)*100/(255*2)
        err_b = math.sqrt(err_b)*100/(255*2)
        return err_r, err_g, err_b

        # ============================ Element Refinement ============================

    def interpolate_rgb(v1, v2, elem_index):
        """Interpolate RGB between two vertices v1, v2 on element edge."""
        elem = elements[elem_index - 1]
        v1 = vertexes[v1 - 1]
        v2 = vertexes[v2 - 1]
        width = abs(vertexes[elem['dr'] - 1]['x'] - vertexes[elem['dl'] - 1]['x'])
        height = abs(vertexes[elem['ul'] - 1]['y'] - vertexes[elem['dl'] - 1]['y'])
        x = math.floor((v1['x'] + v2['x']) / 2)
        y = math.floor((v1['y'] + v2['y']) / 2)
        xx = x - vertexes[elem['dl'] - 1]['x']
        yy = y - vertexes[elem['dl'] - 1]['y']
        r, g, b = inpoint(xx, yy, width, height, elem)
        return create_vertex_rgb(x, y, r, g, b)

    def break_neighbours(index):
        """Ensure neighbouring elements are subdivided appropriately."""
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

    def break_element(index):
        """Subdivide one element into 4 smaller ones."""
        nonlocal total_elements, total_vertexes
        element = elements[index - 1]
        if not element['active']:
            print("error: inactive element split!")
            return

        # Ensure neighbours are consistent
        break_neighbours(index)
        element = elements[index - 1]

        v1 = element['dl']
        v2 = element['ul']
        v3 = element['dr']
        v4 = element['ur']

        v5 = v6 = v7 = v8 = v9 = 0

        # Left edge midpoint
        if element['eld'] != 0:
            v6 = elements[element['eld'] - 1]['ur']
            vert_update(v6)
        else:
            v6 = interpolate_rgb(v1, v2, index)
        if element['elu'] == 0:
            vert_update(v6)

        # Right edge midpoint
        if element['erd'] != 0:
            v8 = elements[element['erd'] - 1]['ul']
            vert_update(v8)
        else:
            v8 = interpolate_rgb(v3, v4, index)
        if element['eru'] == 0:
            vert_update(v8)

        # Top edge midpoint
        if element['eur'] != 0:
            v9 = elements[element['eur'] - 1]['dl']
            vert_update(v9)
        else:
            v9 = interpolate_rgb(v2, v4, index)
        if element['eul'] == 0:
            vert_update(v9)

        # Bottom edge midpoint
        if element['edr'] != 0:
            v5 = elements[element['edr'] - 1]['ul']
            vert_update(v5)
        else:
            v5 = interpolate_rgb(v1, v3, index)
        if element['edl'] == 0:
            vert_update(v5)

        # Center point
        v7 = interpolate_rgb(v6, v8, index)

        # Create 4 new sub-elements
        e1 = create_element(v1, v6, v5, v7)
        e2 = create_element(v6, v2, v7, v9)
        e3 = create_element(v5, v7, v3, v8)
        e4 = create_element(v7, v9, v8, v4)

        # Deactivate parent
        element['active'] = False
        elements[index - 1] = element


    # ============================ Main ============================

    init_mesh()

    redo_error_test = True
    refinement_level = 0
    while redo_error_test and refinement_level < max_refinement_level:
        redo_error_test = False
        for i in range(1, total_elements + 1):
            elem = elements[i - 1]
            if elem['active']:
                er, eg, eb = estimate_error(i)
                if er >= maxerror or eg >= maxerror or eb >= maxerror:
                    redo_error_test = True
                    break_element(i)

        refinement_level += 1

    # interpolate all active elements
    for i in range(total_elements):
        if elements[i]['active']:
            interpolate_elem(elements[i], color_edges_black)

    RGB = np.zeros_like(XX)
    RGB[:, :, 0] = RR
    RGB[:, :, 1] = GG
    RGB[:, :, 2] = BB
    RGB = np.clip(RGB, 0, 255).astype(np.uint8)

    plt.imshow(RGB)
    plt.axis("off")
    plt.show()

    print(f"Total elements: {total_elements}")
    return RGB, elements, vertexes


if __name__ == "__main__" :
    bitmap_h("ziemia.jpg",4,4,1,10,False)