import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

# This is a implementation of h-adaptive bitmap projection.
#
# How to use
# 
# bitmap(filename as a string, number of elements along x axis, number of elements along y axis, maximum relative error, maximum level of adaptivity, if we need to see how elements were broken)
#
# Examples
# 
# bitmap_h("mp.JPG", 10, 10, 0.1, 3, False)
# bitmap_h("basket.JPG", 20, 20, 0.5, 1, True)

def bitmap_h(filename,elementsx,elementsy,maxerror,max_refinement_level,color_edges_black):
    # read image from file
    XX = np.array(Image.open(filename).convert("RGB"))

    # exctract red, green and blue components
    RR = XX[:, :, 0].astype(int)
    GG = XX[:, :, 1].astype(int)
    BB = XX[:, :, 2].astype(int)

    # read size of image
    ix = XX.shape[0]
    iy = XX.shape[1]
    total_vertexes = 0
    total_elements = 0

    # element structure contains:
    # * vertices - organized as followed:
    #
    # ul - ur 
    # |    |
    # dl - dr
    #
    # ul - up-left
    # ur - up-right
    # dl - down-left
    # dr - down-right
    #
    # * neighbours (elements) organized as followed
    # there can be up to two neighbours on each edge
    # default neighbours (if there is one neighbour) are:
    # eul, elu, eru, edl
    #
    #      eul eur
    #      ___ ___
    # elu |       | eru
    #     |       |
    # eld |___ ___| erd
    #      edl edr
    #
    # eul - element-up-left
    # eur - element-up-right
    # elu - element-left-up
    # eld - element-left-down
    # eru - element-right-up
    # erd - element-right-down
    # edl - element-down-left
    # edr - element-down-right
    #
    # * active - we don't delete inactive elements, rather tag them as inactive
    # * index - index of element in global elements table

    elements = [None]

    # vertex data structure contains
    # * x and y coordinates
    # * r, g, b - red, green and blue components
    # * index - index of vertex in global vertexes table
    # * real - False if vertex is hanging node and has interpolated r, g, b components

    vertexes = [None]

    # create vertex (non hanging node)
    def create_vertex(x,y):
        nonlocal total_vertexes, vertexes, RR, GG, BB
        vert = {}
        vert['x'] = int(x)
        vert['y'] = int(y)
        vert['r'] = int(RR[vert['x'], vert['y']])
        vert['g'] = int(GG[vert['x'], vert['y']])
        vert['b'] = int(BB[vert['x'], vert['y']])
        total_vertexes = total_vertexes + 1
        vert['index'] = total_vertexes
        vert['real'] = True
        vertexes.append(vert)
        return total_vertexes

    # create vertex (hanging node)
    def create_vertex_rgb(x,y,r,g,b):
        nonlocal total_vertexes, vertexes
        vert = {}
        vert['x'] = int(x)
        vert['y'] = int(y)
        vert['r'] = int(r)
        vert['g'] = int(g)
        vert['b'] = int(b)
        total_vertexes = total_vertexes + 1
        vert['index'] = total_vertexes
        vert['real'] = False
        vertexes.append(vert)
        return total_vertexes

    # update vertex - when hanging node becomes non-hanging node
    def vert_update(index):
        nonlocal vertexes, RR, GG, BB
        vert = vertexes[index]
        vert['r'] = int(RR[vert['x'], vert['y']])
        vert['g'] = int(GG[vert['x'], vert['y']])
        vert['b'] = int(BB[vert['x'], vert['y']])
        vert['real'] = True
        vertexes[index] = vert

    # create initial element
    def create_element(v1,v2,v3,v4):
        nonlocal total_elements, elements
        element = {}
        element['dl'] = v1
        element['ul'] = v2
        element['dr'] = v3
        element['ur'] = v4
        element['active'] = True
        element['elu'] = 0
        element['eld'] = 0
        element['edl'] = 0
        element['edr'] = 0
        element['eul'] = 0
        element['eur'] = 0
        element['eru'] = 0
        element['erd'] = 0
        total_elements = total_elements + 1
        element['index'] = total_elements
        elements.append(element)
        return element

    # initialize mesh
    def init_mesh():
        # vertexes mapping
        #     
        # v2 -> ul 
        # v4 -> ur
        # v1 -> dl
        # v3 -> dr

        nonlocal ix, iy, elementsx, elementsy, vertexes, elements
        elem_width = ix // elementsx
        elem_hight = iy // elementsy
        # create all vertexes  
        for i in range(elementsy):
            for j in range(elementsx):
                create_vertex(j * elem_width, i * elem_hight)
            create_vertex(ix - 1, i * elem_hight)
        for j in range(elementsx):
            create_vertex(j * elem_width, iy - 1)
        create_vertex(ix - 1, iy - 1)
        # crete all elements
        for r in range(elementsy):
            for c in range(elementsx):
                v1 = r * (elementsx + 1) + c + 1
                v3 = v1 + 1
                v2 = (r + 1) * (elementsx + 1) + c + 1
                v4 = v2 + 1
                element = create_element(v1, v2, v3, v4)
                index = element['index']
                # set neighbours for each element
                if c != 0:
                    element['elu'] = index - 1
                if c != elementsx - 1:
                    element['eru'] = index + 1
                if r != 0:
                    element['edl'] = index - elementsx
                if r != elementsy - 1:
                    element['eul'] = index + elementsx
                elements[index] = element

    # computes r,g,b components of element in given point
    def inpoint(xx,yy,width,hight,elem):
        def fi1(xx,yy):
            x = xx / width if width != 0 else 0
            y = yy / hight if hight != 0 else 0
            return (1 - x) * (1 - y)
        def fi2(xx,yy):
            x = xx / width if width != 0 else 0
            y = yy / hight if hight != 0 else 0
            return (1 - x) * y
        def fi3(xx,yy):
            x = xx / width if width != 0 else 0
            y = yy / hight if hight != 0 else 0
            return x * (1 - y)
        def fi4(xx,yy):
            x = xx / width if width != 0 else 0
            y = yy / hight if hight != 0 else 0
            return x * y
        f1 = fi1(xx,yy)
        f2 = fi2(xx,yy)
        f3 = fi3(xx,yy)
        f4 = fi4(xx,yy)
        r = vertexes[elem['dl']]['r'] * f1
        r = r + vertexes[elem['ul']]['r'] * f2
        r = r + vertexes[elem['dr']]['r'] * f3
        r = r + vertexes[elem['ur']]['r'] * f4
        r = math.floor(r)
        g = vertexes[elem['dl']]['g'] * f1
        g = g + vertexes[elem['ul']]['g'] * f2
        g = g + vertexes[elem['dr']]['g'] * f3
        g = g + vertexes[elem['ur']]['g'] * f4
        g = math.floor(g)
        b = vertexes[elem['dl']]['b'] * f1
        b = b + vertexes[elem['ul']]['b'] * f2
        b = b + vertexes[elem['dr']]['b'] * f3
        b = b + vertexes[elem['ur']]['b'] * f4
        b = math.floor(b)
        return int(r), int(g), int(b)

    # interpolate r, g, b components for hanging node
    # v1 and v2 are vertexes of given element on edges of broken edge
    # v3 is interpolated vertex between v1 and v2
    def interpolate_rgb(v1,v2,element_index):
        nonlocal total_vertexes, vertexes, elements
        elem = elements[element_index]
        width = vertexes[elem['dr']]['x'] - vertexes[elem['dl']]['x']
        hight = vertexes[elem['ul']]['y'] - vertexes[elem['dl']]['y']
        vert1 = vertexes[v1]
        vert2 = vertexes[v2]
        vert3 = {}
        vert3['x'] = math.floor((vert1['x'] + vert2['x']) / 2)
        vert3['y'] = math.floor((vert1['y'] + vert2['y']) / 2)
        xx = vert3['x'] - vertexes[elem['dl']]['x']
        yy = vert3['y'] - vertexes[elem['dl']]['y']
        r,g,b = inpoint(xx,yy,abs(width),abs(hight),elem)
        vert3['r'] = r
        vert3['g'] = g
        vert3['b'] = b
        vert3['real'] = False
        total_vertexes = total_vertexes + 1
        vert3['index'] = total_vertexes
        vertexes.append(vert3)
        return total_vertexes

    # interpolate r,g,b components of a element
    def interpolate_elem(element_index,color_edges_black_flag):
        nonlocal RR, GG, BB, vertexes, elements
        elem = elements[element_index]
        width = vertexes[elem['dr']]['x'] - vertexes[elem['dl']]['x']
        hight = vertexes[elem['ul']]['y'] - vertexes[elem['dl']]['y']
        width = abs(width)
        hight = abs(hight)
        dlx = vertexes[elem['dl']]['x']
        dly = vertexes[elem['dl']]['y']
        for xx in range(0, width + 1):
            for yy in range(0, hight + 1):
                r,g,b = inpoint(xx,yy,width,hight,elem)
                RR[dlx + xx, dly + yy] = r
                GG[dlx + xx, dly + yy] = g
                BB[dlx + xx, dly + yy] = b
        # create black edges on element if requested
        if color_edges_black_flag:
            for xx in range(0, width + 1):
                RR[dlx + xx, dly] = 0
                GG[dlx + xx, dly] = 0
                BB[dlx + xx, dly] = 0
                RR[dlx + xx, dly + hight] = 0
                GG[dlx + xx, dly + hight] = 0
                BB[dlx + xx, dly + hight] = 0
            for yy in range(0, hight + 1):
                RR[dlx, dly + yy] = 0
                GG[dlx, dly + yy] = 0
                BB[dlx, dly + yy] = 0
                RR[dlx + width, dly + yy] = 0
                GG[dlx + width, dly + yy] = 0
                BB[dlx + width, dly + yy] = 0

    # if neighbour is already bigger than element that we try to break - we should break it as well
    def break_neighbours(index):
        nonlocal elements
        element = elements[index]
        def check_left():
            # no neighbours on the left
            if element['elu'] == 0:
                return
            # two neighbours on the left  
            if element['eld'] != 0:
                return
            # only one neighbour on the left
            left = elements[element['elu']]
            # neighbour on the left has two neighbours on the right 
            if left['erd'] != 0:
                break_element(element['elu'])
        def check_right():
            # no neighbours on the right
            if element['eru'] == 0:
                return
            # two neighbours on the right    
            if element['erd'] != 0:
                return
            # only one neighbour on the right
            right = elements[element['eru']]
            if right['eld'] != 0:
                # neighbour on the right has two neighbours on the left
                break_element(element['eru'])
        def check_up():
            # no neighbours on the top
            if element['eul'] == 0:
                return
            # two neighbours on the top  
            if element['eur'] != 0:
                return
            # only one neighbour on the top
            up = elements[element['eul']]
            if up['edr'] != 0:
                # neighbour on the top has two neighbours on the bottom
                break_element(element['eul'])
        def check_down():
            # no neighbours on the bottom
            if element['edl'] == 0:
                return
            # two neighbours on the bottom   
            if element['edr'] != 0:
                return
            # only one neighbour on the bottom
            down = elements[element['edl']]
            if down['eur'] != 0:
                # neighbour on the bottom has two neighbours on the top
                break_element(element['edl'])
        check_left()
        check_right()
        check_up()
        check_down()

    # breaking element
    def break_element(index):
        nonlocal elements, vertexes
        element = elements[index]
        if not element['active']:
            print('error!!!')
        break_neighbours(index)
        element = elements[index]

        # vertexes of element are organized as followed
        #
        # ul - ur 
        # |    |
        # dl - dr
        #
        # they are mapped to local vertices 
        #
        # v2 - v4 
        # |  e  |
        # v1 - v3
        #
        # after breaking element vertices and new elements are organized as followed
        #
        # v2 - v9 - v4
        #  | e2 | e4 |
        # v6 - v7 - v8
        #  | e1 | e3 |
        # v1 - v5 - v3
        #
        # e  -> e2 e4
        #       e1 e3

        v1 = element['dl']
        v2 = element['ul']
        v3 = element['dr']
        v4 = element['ur']
        v5 = 0
        v6 = 0
        v7 = 0
        v8 = 0
        v9 = 0
        # if we have two neighbours left
        if element['eld'] != 0:
            eld = elements[element['eld']]
            v6 = eld['ur']
            vert_update(v6)
        # if we have unbroken neighbour left
        else:
            v6 = interpolate_rgb(v1,v2,index)
        if element['elu'] == 0:
            vert_update(v6)
        # if we have two neighbours right 
        if element['erd'] != 0:
            erd = elements[element['erd']]
            v8 = erd['ul']
            vert_update(v8)
        # if we have unbroken neighbour right
        else:
            v8 = interpolate_rgb(v3,v4,index)
        # if we have two neighbours up  
        if element['eru'] == 0:
            vert_update(v8)
        # if we have unbroken neighbour up
        if element['eur'] != 0:
            eur = elements[element['eur']]
            v9 = eur['dl']
            vert_update(v9)
        else:
            v9 = interpolate_rgb(v2,v4,index)
        if element['eul'] == 0:
            vert_update(v9)
        # if we have two neighbours down  
        if element['edr'] != 0:
            edr = elements[element['edr']]
            v5 = edr['ul']
            vert_update(v5)
        # if we have unbroken neighbour down
        else:
            v5 = interpolate_rgb(v1,v3,index)
        if element['edl'] == 0:
            vert_update(v5)
        x = vertexes[v5]['x']
        y = vertexes[v6]['y']
        v7 = create_vertex(x,y)
        element['active'] = False
        elements[element['index']] = element
        e1 = create_element(v1,v6,v5,v7)
        e2 = create_element(v6,v2,v7,v9)
        e3 = create_element(v5,v7,v3,v8)
        e4 = create_element(v7,v9,v8,v4)
        # set neighbours between new elements
        e1['eru'] = e3['index']
        e1['eul'] = e2['index']
        e2['edl'] = e1['index']
        e2['eru'] = e4['index']
        e3['elu'] = e1['index']
        e3['eul'] = e4['index']
        e4['elu'] = e2['index']
        e4['edl'] = e3['index']
        # set neighbours between new and old elements
        e1['edl'] = element['edl']
        if element['edl'] != 0:
            edl = elements[element['edl']]
            edl['eul'] = e1['index']
            elements[edl['index']] = edl
        if element['edr'] != 0:
            e3['edl'] = element['edr']
            edr = elements[element['edr']]
            edr['eul'] = e3['index']
            elements[edr['index']] = edr
        else:
            e3['edl'] = element['edl']
            if element['edl'] != 0:
                edl = elements[element['edl']]
                edl['eur'] = e3['index']
                elements[edl['index']] = edl
        e2['elu'] = element['elu']
        if element['elu'] != 0:
            elu = elements[element['elu']]
            elu['eru'] = e2['index']
            elements[elu['index']] = elu
        if element['eld'] != 0:
            e1['elu'] = element['eld']
            eld = elements[element['eld']]
            eld['eru'] = e1['index']
            elements[eld['index']] = eld
        else:
            e1['elu'] = element['elu']
            if element['elu'] != 0:
                elu = elements[element['elu']]
                elu['erd'] = e1['index']
                elements[elu['index']] = elu
        e2['eul'] = element['eul']
        if element['eul'] != 0:
            eul = elements[element['eul']]
            eul['edl'] = e2['index']
            elements[eul['index']] = eul
        if element['eur'] != 0:
            e4['eul'] = element['eur']
            eur = elements[element['eur']]
            eur['edl'] = e4['index']
            elements[eur['index']] = eur
        else:
            e4['eul'] = element['eul']
            if element['eul'] != 0:
                eul = elements[element['eul']]
                eul['edr'] = e4['index']
                elements[eul['index']] = eul
        e4['eru'] = element['eru']
        if element['eru'] != 0:
            eru = elements[element['eru']]
            eru['elu'] = e4['index']
            elements[eru['index']] = eru
        if element['erd'] != 0:
            e3['eru'] = element['erd']
            erd = elements[element['erd']]
            erd['elu'] = e3['index']
            elements[erd['index']] = erd
        else:
            e3['eru'] = element['eru']
            if element['eru'] != 0:
                eru = elements[element['eru']]
                eru['eld'] = e3['index']
                elements[eru['index']] = eru
        elements[e4['index']] = e4
        elements[e3['index']] = e3
        elements[e2['index']] = e2
        elements[e1['index']] = e1

    # estimate relative error of interpolation over given element
    def estimate_error(index):
        nonlocal elements, vertexes, RR, GG, BB
        element = elements[index]
        dl = element['dl']
        ul = element['ul']
        dr = element['dr']
        ur = element['ur']
        xl = vertexes[dl]['x']
        yd = vertexes[dl]['y']
        xr = vertexes[ur]['x']
        yu = vertexes[ur]['y']
        elementWidth = xr - xl
        elementHeigth = yu - yd
        # interpolate using L2 norm and Gaussian quadrature rule
        x1 = elementWidth / 2.0 - elementWidth / (math.sqrt(3.0) * 2.0)
        x2 = elementWidth / 2.0 + elementWidth / (math.sqrt(3.0) * 2.0)
        y1 = elementHeigth / 2.0 - elementHeigth / (math.sqrt(3.0) * 2.0)
        y2 = elementHeigth / 2.0 + elementHeigth / (math.sqrt(3.0) * 2.0)
        x1 = math.floor(x1)
        x2 = math.floor(x2)
        y1 = math.floor(y1)
        y2 = math.floor(y2)
        r1,g1,b1 = inpoint(x1,y1,elementWidth,elementHeigth,element)
        r2,g2,b2 = inpoint(x1,y2,elementWidth,elementHeigth,element)
        r3,g3,b3 = inpoint(x2,y1,elementWidth,elementHeigth,element)
        r4,g4,b4 = inpoint(x2,y2,elementWidth,elementHeigth,element)
        r1 = r1 - RR[x1 + xl, y1 + yd]
        g1 = g1 - GG[x1 + xl, y1 + yd]
        b1 = b1 - BB[x1 + xl, y1 + yd]
        r2 = r2 - RR[x1 + xl, y2 + yd]
        g2 = g2 - GG[x1 + xl, y2 + yd]
        b2 = b2 - BB[x1 + xl, y2 + yd]
        r3 = r3 - RR[x2 + xl, y1 + yd]
        g3 = g3 - GG[x2 + xl, y1 + yd]
        b3 = b3 - BB[x2 + xl, y1 + yd]
        r4 = r4 - RR[x2 + xl, y2 + yd]
        g4 = g4 - GG[x2 + xl, y2 + yd]
        b4 = b4 - BB[x2 + xl, y2 + yd]
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

    init_mesh()
    redo_error_test = True
    refinemenet_level = 0
    # repeat until we match maximum local estimation error or maximum refinemenet level
    while redo_error_test and (refinemenet_level < max_refinement_level):
        redo_error_test = False
        # loop through elements
        for i in range(1, len(elements)):
            # check only active elements
            if elements[i]['active']:
                # estimate realtive interpolation error in red, green and blue components
                rr,gg,bb = estimate_error(i)
                # if any of the errors is higher than our maximum -> break element and repeat entire loop
                if (rr >= maxerror) or (gg >= maxerror) or (bb >= maxerror):
                    redo_error_test = True
                    break_element(i)
        refinemenet_level = refinemenet_level + 1
    # interpolate all active elements - recreate bitmap red green and blue compoments
    for i in range(1, len(elements)):
        if elements[i]['active']:
            interpolate_elem(i,color_edges_black)
    
    # recreate bitmap from red, green and blue compoments
    RGB = XX.copy()
    RGB[:, :, 0] = RR
    RGB[:, :, 1] = GG
    RGB[:, :, 2] = BB

    # display image
    plt.imshow(RGB.astype(np.uint8))
    plt.axis('off')
    plt.show()
    print(len(elements) - 1)

if __name__ == '__main__':
    bitmap_h("ziemia.jpg", 4,4,1,5,True)
