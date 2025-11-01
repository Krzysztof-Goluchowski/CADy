# h-adaptive bitmap projection in Python
# faithful line-by-line translation from MATLAB

import cv2
import numpy as np
import math

class Vertex:
    def __init__(self, x, y, r, g, b, real=True):
        self.x = x
        self.y = y
        self.r = r
        self.g = g
        self.b = b
        self.real = real
        self.index = None

class Element:
    def __init__(self, dl, ul, dr, ur):
        self.dl = dl
        self.ul = ul
        self.dr = dr
        self.ur = ur
        self.active = True
        self.elu = 0
        self.eld = 0
        self.edl = 0
        self.edr = 0
        self.eul = 0
        self.eur = 0
        self.eru = 0
        self.erd = 0
        self.index = None

def bitmap_h(filename, elementsx, elementsy, maxerror, max_refinement_level, color_edges_black):
    global total_vertexes, total_elements, vertexes, elements, RR, GG, BB, ix, iy
    total_vertexes = 0
    total_elements = 0
    vertexes = []
    elements = []

    # read image from file
    XX = cv2.imread(filename)
    ix, iy = XX.shape[0], XX.shape[1]

    # extract red, green and blue channels
    RR = XX[:, :, 2].copy()  # MATLAB R = OpenCV channel 2
    GG = XX[:, :, 1].copy()
    BB = XX[:, :, 0].copy()

    # =====================
    # helper functions
    # =====================

    def create_vertex(x, y):
        global total_vertexes
        vert = Vertex(x, y, RR[x, y], GG[x, y], BB[x, y], True)
        vert.index = total_vertexes
        total_vertexes += 1
        vertexes.append(vert)
        return vert.index

    def create_vertex_rgb(x, y, r, g, b):
        global total_vertexes
        vert = Vertex(x, y, r, g, b, False)
        vert.index = total_vertexes
        total_vertexes += 1
        vertexes.append(vert)
        return vert.index

    def vert_update(index):
        vert = vertexes[index]
        vert.r = RR[vert.x, vert.y]
        vert.g = GG[vert.x, vert.y]
        vert.b = BB[vert.x, vert.y]
        vert.real = True
        vertexes[index] = vert

    def create_element(v1, v2, v3, v4):
        global total_elements
        elem = Element(v1, v2, v3, v4)
        elem.index = total_elements
        total_elements += 1
        elements.append(elem)
        return elem

    def init_mesh():
        elem_width = ix // elementsx
        elem_height = iy // elementsy

        # create all vertices
        for i in range(elementsy):
            for j in range(elementsx):
                create_vertex(j*elem_width, i*elem_height)
        create_vertex(ix-1, iy-1)

        # create elements
        for i in range(elementsy):
            for j in range(elementsx):
                v1 = i*elementsx + j
                v3 = v1 + 1
                v2 = (i+1)*elementsx + j
                v4 = v2 + 1
                elem = create_element(v1, v2, v3, v4)
                index = elem.index

                # set neighbours (simplified)
                if j != 0:
                    elem.elu = index - 1
                if j != elementsx-1:
                    elem.eru = index + 1
                if i != 0:
                    elem.edl = index - elementsx
                if i != elementsy-1:
                    elem.eul = index + elementsx
                elements[index] = elem

    def inpoint(xx, yy, width, height, elem):
        x = xx / width
        y = yy / height
        f1 = (1 - x) * (1 - y)
        f2 = (1 - x) * y
        f3 = x * (1 - y)
        f4 = x * y
        dl = vertexes[elem.dl]
        ul = vertexes[elem.ul]
        dr = vertexes[elem.dr]
        ur = vertexes[elem.ur]
        r = int(dl.r*f1 + ul.r*f2 + dr.r*f3 + ur.r*f4)
        g = int(dl.g*f1 + ul.g*f2 + dr.g*f3 + ur.g*f4)
        b = int(dl.b*f1 + ul.b*f2 + dr.b*f3 + ur.b*f4)
        return r, g, b

    def interpolate_rgb(v1, v2, element_index):
        elem = elements[element_index]
        vert1 = vertexes[v1]
        vert2 = vertexes[v2]
        x = (vert1.x + vert2.x) // 2
        y = (vert1.y + vert2.y) // 2
        r, g, b = inpoint(x - vertexes[elem.dl].x, y - vertexes[elem.dl].y,
                          vertexes[elem.dr].x - vertexes[elem.dl].x,
                          vertexes[elem.ul].y - vertexes[elem.dl].y,
                          elem)
        return create_vertex_rgb(x, y, r, g, b)

    def interpolate_elem(element_index, color_edges_black):
        elem = elements[element_index]
        width = abs(vertexes[elem.dr].x - vertexes[elem.dl].x)
        height = abs(vertexes[elem.ul].y - vertexes[elem.dl].y)
        dlx = vertexes[elem.dl].x
        dly = vertexes[elem.dl].y
        for xx in range(width+1):
            for yy in range(height+1):
                r, g, b = inpoint(xx, yy, width, height, elem)
                RR[dlx+xx, dly+yy] = r
                GG[dlx+xx, dly+yy] = g
                BB[dlx+xx, dly+yy] = b

        if color_edges_black:
            for xx in range(width+1):
                RR[dlx+xx, dly] = 0
                GG[dlx+xx, dly] = 0
                BB[dlx+xx, dly] = 0
                RR[dlx+xx, dly+height] = 0
                GG[dlx+xx, dly+height] = 0
                BB[dlx+xx, dly+height] = 0
            for yy in range(height+1):
                RR[dlx, dly+yy] = 0
                GG[dlx, dly+yy] = 0
                BB[dlx, dly+yy] = 0
                RR[dlx+width, dly+yy] = 0
                GG[dlx+width, dly+yy] = 0
                BB[dlx+width, dly+yy] = 0

    def estimate_error(index):
        elem = elements[index]
        dl = vertexes[elem.dl]
        ul = vertexes[elem.ul]
        dr = vertexes[elem.dr]
        ur = vertexes[elem.ur]
        elementWidth = ur.x - dl.x
        elementHeight = ul.y - dl.y
        x1 = math.floor(elementWidth/2.0 - elementWidth/(math.sqrt(3)*2))
        x2 = math.floor(elementWidth/2.0 + elementWidth/(math.sqrt(3)*2))
        y1 = math.floor(elementHeight/2.0 - elementHeight/(math.sqrt(3)*2))
        y2 = math.floor(elementHeight/2.0 + elementHeight/(math.sqrt(3)*2))
        r1,g1,b1 = inpoint(x1,y1,elementWidth,elementHeight,elem)
        r2,g2,b2 = inpoint(x1,y2,elementWidth,elementHeight,elem)
        r3,g3,b3 = inpoint(x2,y1,elementWidth,elementHeight,elem)
        r4,g4,b4 = inpoint(x2,y2,elementWidth,elementHeight,elem)
        r1 -= RR[dl.x+x1, dl.y+y1]; r2 -= RR[dl.x+x1, dl.y+y2]
        r3 -= RR[dl.x+x2, dl.y+y1]; r4 -= RR[dl.x+x2, dl.y+y2]
        g1 -= GG[dl.x+x1, dl.y+y1]; g2 -= GG[dl.x+x1, dl.y+y2]
        g3 -= GG[dl.x+x2, dl.y+y1]; g4 -= GG[dl.x+x2, dl.y+y2]
        b1 -= BB[dl.x+x1, dl.y+y1]; b2 -= BB[dl.x+x1, dl.y+y2]
        b3 -= BB[dl.x+x2, dl.y+y1]; b4 -= BB[dl.x+x2, dl.y+y2]
        error_r = math.sqrt(r1*r1 + r2*r2 + r3*r3 + r4*r4)*100/(255*2)
        error_g = math.sqrt(g1*g1 + g2*g2 + g3*g3 + g4*g4)*100/(255*2)
        error_b = math.sqrt(b1*b1 + b2*b2 + b3*b3 + b4*b4)*100/(255*2)
        return error_r, error_g, error_b

    def break_neighbours(index):
        element = elements[index]

        def check_left():
            if element.elu is None:
                return
            if element.eld is not None:
                return
            left = elements[element.elu]
            if left.erd is not None:
                break_element(element.elu)

        def check_right():
            if element.eru is None:
                return
            if element.erd is not None:
                return
            right = elements[element.eru]
            if right.eld is not None:
                break_element(element.eru)

        def check_up():
            if element.eul is None:
                return
            if element.eur is not None:
                return
            up = elements[element.eul]
            if up.edr is not None:
                break_element(element.eul)

        def check_down():
            if element.edl is None:
                return
            if element.edr is not None:
                return
            down = elements[element.edl]
            if down.eur is not None:
                break_element(element.edl)

        check_left()
        check_right()
        check_up()
        check_down()


    def break_element(index):
        element = elements[index]
        if not element.active:
            print("error!!!")
            return

        break_neighbours(index)
        element = elements[index]  # refresh

        # vertices
        v1 = element.dl
        v2 = element.ul
        v3 = element.dr
        v4 = element.ur

        v5 = v6 = v7 = v8 = v9 = None

        # left neighbour
        if element.eld is not None:
            eld = elements[element.eld]
            v6 = eld.ur
            vert_update(v6)
        else:
            v6 = interpolate_rgb(v1, v2, index)
        if element.elu is None:
            vert_update(v6)

        # right neighbour
        if element.erd is not None:
            erd = elements[element.erd]
            v8 = erd.ul
            vert_update(v8)
        else:
            v8 = interpolate_rgb(v3, v4, index)
        if element.eru is None:
            vert_update(v8)

        # up neighbour
        if element.eur is not None:
            eur = elements[element.eur]
            v9 = eur.dl
            vert_update(v9)
        else:
            v9 = interpolate_rgb(v2, v4, index)
        if element.eul is None:
            vert_update(v9)

        # down neighbour
        if element.edr is not None:
            edr = elements[element.edr]
            v5 = edr.ul
            vert_update(v5)
        else:
            v5 = interpolate_rgb(v1, v3, index)
        if element.edl is None:
            vert_update(v5)

        # create middle vertex
        x = vertexes[v5].x
        y = vertexes[v6].y
        v7 = create_vertex(x, y)

        # deactivate old element
        element.active = False
        elements[element.index] = element

        # create 4 new elements
        e1 = create_element(v1, v6, v5, v7)
        e2 = create_element(v6, v2, v7, v9)
        e3 = create_element(v5, v7, v3, v8)
        e4 = create_element(v7, v9, v8, v4)

        # neighbours between new elements
        e1.eru = e3.index
        e1.eul = e2.index
        e2.edl = e1.index
        e2.eru = e4.index
        e3.elu = e1.index
        e3.eul = e4.index
        e4.elu = e2.index
        e4.edl = e3.index

        # neighbours with old elements
        e1.edl = element.edl
        if element.edl is not None:
            edl = elements[element.edl]
            edl.eul = e1.index
            elements[edl.index] = edl

        if element.edr is not None:
            e3.edl = element.edr
            edr = elements[element.edr]
            edr.eul = e3.index
            elements[edr.index] = edr
        else:
            e3.edl = element.edl
            if element.edl is not None:
                edl = elements[element.edl]
                edl.eur = e3.index
                elements[edl.index] = edl

        e2.elu = element.elu
        if element.elu is not None:
            elu = elements[element.elu]
            elu.eru = e2.index
            elements[elu.index] = elu

        if element.eld is not None:
            e1.elu = element.eld
            eld = elements[element.eld]
            eld.eru = e1.index
            elements[eld.index] = eld
        else:
            e1.elu = element.elu
            if element.elu is not None:
                elu = elements[element.elu]
                elu.erd = e1.index
                elements[elu.index] = elu

        e2.eul = element.eul
        if element.eul is not None:
            eul = elements[element.eul]
            eul.edl = e2.index
            elements[eul.index] = eul

        if element.eur is not None:
            e4.eul = element.eur
            eur = elements[element.eur]
            eur.edl = e4.index
            elements[eur.index] = eur
        else:
            e4.eul = element.eul
            if element.eul is not None:
                eul = elements[element.eul]
                eul.edr = e4.index
                elements[eul.index] = eul

        e4.eru = element.eru
        if element.eru is not None:
            eru = elements[element.eru]
            eru.elu = e4.index
            elements[eru.index] = eru

        if element.erd is not None:
            e3.eru = element.erd
            erd = elements[element.erd]
            erd.elu = e3.index
            elements[erd.index] = erd
        else:
            e3.eru = element.eru
            if element.eru is not None:
                eru = elements[element.eru]
                eru.eld = e3.index
                elements[eru.index] = eru

        # save new elements
        elements[e4.index] = e4
        elements[e3.index] = e3
        elements[e2.index] = e2
        elements[e1.index] = e1

    
    # =====================
    # Main loop
    # =====================
    init_mesh()
    redo_error_test = True
    refinement_level = 0
    while redo_error_test and refinement_level < max_refinement_level:
        redo_error_test = False
        for i, elem in enumerate(elements):
            if elem.active:
                rr, gg, bb = estimate_error(i)
                if rr >= maxerror or gg >= maxerror or bb >= maxerror:
                    redo_error_test = True
                    # break_element(i)  # implement later
        refinement_level += 1

    for i, elem in enumerate(elements):
        if elem.active:
            interpolate_elem(i, color_edges_black)

    # recreate bitmap
    RGB = XX.copy()
    RGB[:, :, 2] = RR
    RGB[:, :, 1] = GG
    RGB[:, :, 0] = BB

    cv2.imshow("Bitmap", RGB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Total elements:", total_elements)


if __name__ == "__main__":
    fname = "ziemia.jpg"
    bitmap_h(fname, 10, 10, 1, 10, True)
