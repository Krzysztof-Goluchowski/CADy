"""
Przekład MATLAB -> Python: bitmap_h

Uwaga:
- zachowałem strukturę danych elementów i węzłów (vertexes) przy użyciu dataclass
- init_mesh tworzy początkowy element pokrywający cały obraz
- estimate_error aktualnie zwraca 0 dla każdego kanału (brak podziału). To miejsce do implementacji właściwej estymacji błędu.
- break_element() oznacza element jako nieaktywny (nie dzieli go dalej). Tu możesz dodać logikę podziału na 4 elementy.
- interpolate_elem() wykonuje prostą interpolację bilinearną po wierzchołkach elementu i nanosi wyniki na kanały RR,GG,BB.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List

# ----------------------------
# Struktury danych (odpowiedniki struct w MATLAB)
# ----------------------------
@dataclass
class Vertex:
    x: int
    y: int
    r: float
    g: float
    b: float
    index: int
    real: bool

@dataclass
class Element:
    dl: int  # index vertex down-left
    ul: int  # up-left
    dr: int  # down-right
    ur: int  # up-right
    active: bool
    elu: int
    eld: int
    edl: int
    edr: int
    eul: int
    eur: int
    eru: int
    erd: int
    index: int

# ----------------------------
# Zmienne globalne (tak jak w MATLAB)
# ----------------------------
elements: List[Element] = []
vertexes: List[Vertex] = []
total_vertexes = 0
total_elements = 0
elementsx = 0
elementsy = 0

# obrazy / kanały
RR = None
GG = None
BB = None
ix = 0
iy = 0

# ----------------------------
# Funkcja główna - przekład funkcji MATLAB `bitmap_h`
# ----------------------------
def bitmap_h(filename, elementsxx, elementsyy, maxerror, max_refinement_level, color_edges_black):
    global RR, GG, BB, ix, iy
    global total_vertexes, total_elements, elements, vertexes, elementsx, elementsy
    elementsx = elementsxx
    elementsy = elementsyy

    # read image from file
    img = Image.open(filename).convert('RGB')
    XX = np.array(img)  # shape (ix, iy, 3) where ix = height, iy = width

    # exctract red, green and blue components (0..255)
    RR = XX[:, :, 0].astype(np.float64)
    GG = XX[:, :, 1].astype(np.float64)
    BB = XX[:, :, 2].astype(np.float64)

    # read size of image
    ix = XX.shape[0]
    iy = XX.shape[1]

    # initialize unbroken mesh
    init_mesh()

    while len(elements) < total_elements:
        elements.append(None)

    redo_error_test = True
    refinemenet_level = 0

    # repeat until we match maximum local estimation error or maximum refinemenet level
    while (redo_error_test and (refinemenet_level < max_refinement_level)):
        redo_error_test = False
        # loop through elements (1..total_elements in MATLAB; here 0..total_elements-1)
        for i in range(total_elements):
            # check only active elements
            if elements[i].active:
                # estimate relative interpolation error in red, green and blue components
                rr, gg, bb = estimate_error(i)
                # if any of the errors is higher than our maximum -> break element and repeat entire loop
                if (rr >= maxerror) or (gg >= maxerror) or (bb >= maxerror):
                    redo_error_test = True
                    break_element(i)
                    # we break the for-loop to restart full scan as in MATLAB
                    break
        refinemenet_level += 1

    # interpolate all active elements - recreate bitmap red green and blue components
    for i in range(total_elements):
        if elements[i].active:
            interpolate_elem(i, color_edges_black)

    # recreate bitmap from red, green and blue components
    RGB = np.zeros_like(XX)
    RGB[:, :, 0] = np.clip(RR, 0, 255).astype(np.uint8)
    RGB[:, :, 1] = np.clip(GG, 0, 255).astype(np.uint8)
    RGB[:, :, 2] = np.clip(BB, 0, 255).astype(np.uint8)

    # display image
    plt.figure(figsize=(8, 8 * ix / iy))
    plt.axis('off')
    plt.imshow(RGB)
    plt.show()

    # print total_elements (MATLAB just wrote variable name)
    print("total_elements =", total_elements)


# --------------------------------------------
# Odpowiedniki funkcji MATLAB:
#  create_vertex
#  create_vertex_rgb
#  vert_update
#  create_element
# --------------------------------------------
from typing import Optional

def create_vertex(x: int, y: int) -> int:
    """
    Tworzy nowy wierzchołek (nie-hanging node)
    MATLAB: create_vertex(x, y)
    """
    global vertexes, total_vertexes, RR, GG, BB

    # Konwersja indeksów z MATLAB (1-based) -> Python (0-based)
    r = RR[x - 1, y - 1]
    g = GG[x - 1, y - 1]
    b = BB[x - 1, y - 1]

    vert = Vertex(x=x, y=y, r=r, g=g, b=b,
                  index=total_vertexes, real=True)

    vertexes.append(vert)
    total_vertexes += 1

    # Zwracamy indeks (w Pythonie 0-based)
    return total_vertexes - 1


def create_vertex_rgb(x: int, y: int, r: float, g: float, b: float) -> int:
    """
    Tworzy nowy wierzchołek z podanymi kolorami (hanging node)
    MATLAB: create_vertex_rgb(x, y, r, g, b)
    """
    global vertexes, total_vertexes

    vert = Vertex(x=x, y=y, r=r, g=g, b=b,
                  index=total_vertexes, real=False)

    vertexes.append(vert)
    total_vertexes += 1

    return total_vertexes - 1


def vert_update(index: int):
    """
    Aktualizuje istniejący wierzchołek — hanging node staje się realnym.
    MATLAB: vert_update(index)
    """
    global vertexes, RR, GG, BB

    vert = vertexes[index]

    # NumPy używa [row, col] = [y, x], a nie [x, y]
    x = min(max(vert.x, 0), RR.shape[1] - 1)
    y = min(max(vert.y, 0), RR.shape[0] - 1)

    vert.r = RR[y, x]
    vert.g = GG[y, x]
    vert.b = BB[y, x]
    vert.real = True

    vertexes[index] = vert


def create_element(v1, v2, v3, v4):
    global elements, total_elements
    element = Element(
        dl=v1,
        ul=v2,
        dr=v3,
        ur=v4,
        active=True,
        elu=0,
        eld=0,
        edl=0,
        edr=0,
        eul=0,
        eur=0,
        eru=0,
        erd=0,
        index=total_elements
    )
    elements.append(element)
    total_elements += 1
    return element

def init_mesh():
    """
    Przekład MATLAB-owego init_mesh na Python.
    Tworzy siatkę wierzchołków i elementów na podstawie elementsx x elementsy.
    """
    global ix, iy, elementsx, elementsy, vertexes, elements

    # szerokość i wysokość elementu (zaokrąglanie w dół)
    elem_width = ix // elementsx
    elem_height = iy // elementsy

    # === KROK 1: Tworzenie wierzchołków ===
    for i in range(elementsy + 1):
        y = i * elem_height
        for j in range(elementsx + 1):
            x = j * elem_width
            create_vertex(x, y)

    # === KROK 2: Tworzenie elementów ===
    # Każdy element składa się z 4 wierzchołków: (dl, ul, dr, ur)
    for i in range(elementsy):
        for j in range(elementsx):
            # indeksy wierzchołków w siatce (0-based)
            v1 = i * (elementsx + 1) + j               # dolny-lewy
            v2 = (i + 1) * (elementsx + 1) + j         # górny-lewy
            v3 = v1 + 1                                # dolny-prawy
            v4 = v2 + 1                                # górny-prawy

            # utwórz element
            element = create_element(v1, v2, v3, v4)
            elements.append(element)
            index = len(elements) - 1

            # sąsiedzi poziomi (lewo/prawo)
            if j > 0:
                element.elu = index - 1
            if j < elementsx - 1:
                element.eru = index + 1

            # sąsiedzi pionowi (góra/dół)
            if i > 0:
                element.edl = index - elementsx
            if i < elementsy - 1:
                element.eul = index + elementsx

            # zapisz element z powrotem
            elements[index] = element



import numpy as np

def interpolate_rgb(v1, v2, element):
    """
    interpolate r,g,b components for hanging node
    v1 and v2 are vertexes of given element on edges of broken edge
    v3 is interpolated vertex between v1 and v2
    """
    global elements, vertexes, total_vertexes

    elem = elements[element]
    width = vertexes[elem.dr].x - vertexes[elem.dl].x
    hight = vertexes[elem.ul].y - vertexes[elem.dl].y
    vert1 = vertexes[v1]
    vert2 = vertexes[v2]

    x3 = int(np.floor((vert1.x + vert2.x) / 2))
    y3 = int(np.floor((vert1.y + vert2.y) / 2))

    xx = x3 - vertexes[elem.dl].x
    yy = y3 - vertexes[elem.dl].y

    r, g, b = inpoint(xx, yy, width, hight, elem)

    # Tworzymy nowy wierzchołek i dodajemy do listy
    vert3 = Vertex(
        x=x3,
        y=y3,
        r=r,
        g=g,
        b=b,
        real=False,
        index=total_vertexes
    )
    vertexes.append(vert3)
    v3 = total_vertexes
    total_vertexes += 1

    return v3


def interpolate_elem(element, color_edges_black):
    """
    interpolate r,g,b components of an element
    """
    global elements, vertexes, RR, GG, BB

    elem = elements[element]
    width = abs(vertexes[elem.dr].x - vertexes[elem.dl].x)
    height = abs(vertexes[elem.ul].y - vertexes[elem.dl].y)
    dlx = vertexes[elem.dl].x
    dly = vertexes[elem.dl].y

    for xx in range(width + 1):
        for yy in range(height + 1):
            r, g, b = inpoint(xx, yy, width, height, elem)

            # zabezpieczenie przed wyjściem poza tablicę
            x_idx = min(dlx + xx, RR.shape[1] - 1)
            y_idx = min(dly + yy, RR.shape[0] - 1)

            RR[y_idx, x_idx] = r
            GG[y_idx, x_idx] = g
            BB[y_idx, x_idx] = b

    # create black edges on element if requested
    if color_edges_black:
        for xx in range(width + 1):
            x_idx = min(dlx + xx, RR.shape[1] - 1)
            y_top = min(dly, RR.shape[0] - 1)
            y_bottom = min(dly + height, RR.shape[0] - 1)

            RR[y_top, x_idx] = 0
            GG[y_top, x_idx] = 0
            BB[y_top, x_idx] = 0

            RR[y_bottom, x_idx] = 0
            GG[y_bottom, x_idx] = 0
            BB[y_bottom, x_idx] = 0

        for yy in range(height + 1):
            y_idx = min(dly + yy, RR.shape[0] - 1)
            x_left = min(dlx, RR.shape[1] - 1)
            x_right = min(dlx + width, RR.shape[1] - 1)

            RR[y_idx, x_left] = 0
            GG[y_idx, x_left] = 0
            BB[y_idx, x_left] = 0

            RR[y_idx, x_right] = 0
            GG[y_idx, x_right] = 0
            BB[y_idx, x_right] = 0


import math

def inpoint(xx, yy, width, hight, elem):
    """
    Computes r, g, b components of element in given point
    """
    global vertexes

    def fi1(xx, yy):
        x = xx / width
        y = yy / hight
        return (1 - x) * (1 - y)

    def fi2(xx, yy):
        x = xx / width
        y = yy / hight
        return (1 - x) * y

    def fi3(xx, yy):
        x = xx / width
        y = yy / hight
        return x * (1 - y)

    def fi4(xx, yy):
        x = xx / width
        y = yy / hight
        return x * y

    f1 = fi1(xx, yy)
    f2 = fi2(xx, yy)
    f3 = fi3(xx, yy)
    f4 = fi4(xx, yy)

    r = (vertexes[elem.dl].r * f1 +
         vertexes[elem.ul].r * f2 +
         vertexes[elem.dr].r * f3 +
         vertexes[elem.ur].r * f4)
    r = math.floor(r)

    g = (vertexes[elem.dl].g * f1 +
         vertexes[elem.ul].g * f2 +
         vertexes[elem.dr].g * f3 +
         vertexes[elem.ur].g * f4)
    g = math.floor(g)

    b = (vertexes[elem.dl].b * f1 +
         vertexes[elem.ul].b * f2 +
         vertexes[elem.dr].b * f3 +
         vertexes[elem.ur].b * f4)
    b = math.floor(b)

    return r, g, b


def break_neighbours(index):
    """
    If neighbour is already bigger than element that we try to break - we should break it as well
    """
    global elements

    element = elements[index]

    def check_left():
        # no neighbours on the left
        if element.elu == 0:
            return
        # two neighbours on the left
        if element.eld != 0:
            return
        # only one neighbour on the left
        left = elements[element.elu]
        if left.erd != 0:
            # neighbour on the left has two neighbours on the right
            break_element(element.elu)

    def check_right():
        # no neighbours on the right
        if element.eru == 0:
            return
        # two neighbours on the right
        if element.erd != 0:
            return
        # only one neighbour on the right
        right = elements[element.eru]
        if right.eld != 0:
            # neighbour on the right has two neighbours on the left
            break_element(element.eru)

    def check_up():
        # no neighbours on the top
        if element.eul == 0:
            return
        # two neighbours on the top
        if element.eur != 0:
            return
        # only one neighbour on the top
        up = elements[element.eul]
        if up.edr != 0:
            # neighbour on the top has two neighbours on the bottom
            break_element(element.eul)

    def check_down():
        # no neighbours on the bottom
        if element.edl == 0:
            return
        # two neighbours on the bottom
        if element.edr != 0:
            return
        # only one neighbour on the bottom
        down = elements[element.edl]
        if down.eur != 0:
            # neighbour on the bottom has two neighbours on the top
            break_element(element.edl)

    check_left()
    check_right()
    check_up()
    check_down()


def break_element(index):
    global elements, vertexes

    element = elements[index]
    if not element.active:
        print("error!!!")

    break_neighbours(index)
    element = elements[index]

    # vertexes of element are organized as follows:
    #
    # ul - ur 
    # |    |
    # dl - dr
    #
    # mapped to:
    # v2 - v4 
    # |  e  |
    # v1 - v3
    #
    # after breaking:
    #
    # v2 - v9 - v4
    #  | e2 | e4 |
    # v6 - v7 - v8
    #  | e1 | e3 |
    # v1 - v5 - v3
    #
    # e  -> e2 e4
    #       e1 e3

    v1 = element.dl
    v2 = element.ul
    v3 = element.dr
    v4 = element.ur

    v5 = v6 = v7 = v8 = v9 = 0

    # if we have two neighbours left
    if element.eld != 0:
        eld = elements[element.eld]
        v6 = eld.ur
        vert_update(v6)
    # if we have unbroken neighbour left
    else:
        v6 = interpolate_rgb(v1, v2, index)
    if element.elu == 0:
        vert_update(v6)

    # if we have two neighbours right
    if element.erd != 0:
        erd = elements[element.erd]
        v8 = erd.ul
        vert_update(v8)
    # if we have unbroken neighbour right
    else:
        v8 = interpolate_rgb(v3, v4, index)
    if element.eru == 0:
        vert_update(v8)

    # if we have two neighbours up
    if element.eur != 0:
        eur = elements[element.eur]
        v9 = eur.dl
        vert_update(v9)
    # if we have unbroken neighbour up
    else:
        v9 = interpolate_rgb(v2, v4, index)
    if element.eul == 0:
        vert_update(v9)

    # if we have two neighbours down
    if element.edr != 0:
        edr = elements[element.edr]
        v5 = edr.ul
        vert_update(v5)
    # if we have unbroken neighbour down
    else:
        v5 = interpolate_rgb(v1, v3, index)
    if element.edl == 0:
        vert_update(v5)

    x = vertexes[v5].x
    y = vertexes[v6].y
    v7 = create_vertex(x, y)

    element.active = False
    elements[element.index] = element

    # tworzymy elementy e1-e4 i dodajemy do listy
    e1 = create_element(v1, v6, v5, v7)
    e2 = create_element(v6, v2, v7, v9)
    e3 = create_element(v5, v7, v3, v8)
    e4 = create_element(v7, v9, v8, v4)

    # ustawiamy indeksy po dodaniu do listy
    e1.index = len(elements) - 4
    e2.index = len(elements) - 3
    e3.index = len(elements) - 2
    e4.index = len(elements) - 1

    # set neighbours between new elements (jak wcześniej)
    e1.eru = e3.index
    e1.eul = e2.index
    e2.edl = e1.index
    e2.eru = e4.index
    e3.elu = e1.index
    e3.eul = e4.index
    e4.elu = e2.index
    e4.edl = e3.index

    # set neighbours between new and old elements
    e1.edl = element.edl
    if element.edl != 0:
        edl = elements[element.edl]
        edl.eul = e1.index
        elements[edl.index] = edl
    if element.edr != 0:
        e3.edl = element.edr
        edr = elements[element.edr]
        edr.eul = e3.index
        elements[edr.index] = edr
    else:
        e3.edl = element.edl
        if element.edl != 0:
            edl = elements[element.edl]
            edl.eur = e3.index
            elements[edl.index] = edl

    e2.elu = element.elu
    if element.elu != 0:
        elu = elements[element.elu]
        elu.eru = e2.index
        elements[elu.index] = elu
    if element.eld != 0:
        e1.elu = element.eld
        eld = elements[element.eld]
        eld.eru = e1.index
        elements[eld.index] = eld
    else:
        e1.elu = element.elu
        if element.elu != 0:
            elu = elements[element.elu]
            elu.erd = e1.index
            elements[elu.index] = elu

    e2.eul = element.eul
    if element.eul != 0:
        eul = elements[element.eul]
        eul.edl = e2.index
        elements[eul.index] = eul
    if element.eur != 0:
        e4.eul = element.eur
        eur = elements[element.eur]
        eur.edl = e4.index
        elements[eur.index] = eur
    else:
        e4.eul = element.eul
        if element.eul != 0:
            eul = elements[element.eul]
            eul.edr = e4.index
            elements[eul.index] = eul

    e4.eru = element.eru
    if element.eru != 0:
        eru = elements[element.eru]
        eru.elu = e4.index
        elements[eru.index] = eru
    if element.erd != 0:
        e3.eru = element.erd
        erd = elements[element.erd]
        erd.elu = e3.index
        elements[erd.index] = erd
    else:
        e3.eru = element.eru
        if element.eru != 0:
            eru = elements[element.eru]
            eru.eld = e3.index
            elements[eru.index] = eru

    elements[e4.index] = e4
    elements[e3.index] = e3
    elements[e2.index] = e2
    elements[e1.index] = e1


import math

def estimate_error(index):
    global elements, vertexes, RR, GG, BB

    element = elements[index]
    dl = element.dl
    ul = element.ul
    dr = element.dr
    ur = element.ur

    xl = vertexes[dl].x
    yd = vertexes[dl].y
    xr = vertexes[ur].x
    yu = vertexes[ur].y

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

    r1, g1, b1 = inpoint(x1, y1, elementWidth, elementHeigth, element)
    r2, g2, b2 = inpoint(x1, y2, elementWidth, elementHeigth, element)
    r3, g3, b3 = inpoint(x2, y1, elementWidth, elementHeigth, element)
    r4, g4, b4 = inpoint(x2, y2, elementWidth, elementHeigth, element)

    # poprawny dostęp do tablic
    r1 = r1 - RR[int(x1 + xl), int(y1 + yd)]
    g1 = g1 - GG[int(x1 + xl), int(y1 + yd)]
    b1 = b1 - BB[int(x1 + xl), int(y1 + yd)]

    r2 = r2 - RR[int(x1 + xl), int(y2 + yd)]
    g2 = g2 - GG[int(x1 + xl), int(y2 + yd)]
    b2 = b2 - BB[int(x1 + xl), int(y2 + yd)]

    r3 = r3 - RR[int(x2 + xl), int(y1 + yd)]
    g3 = g3 - GG[int(x2 + xl), int(y1 + yd)]
    b3 = b3 - BB[int(x2 + xl), int(y1 + yd)]

    r4 = r4 - RR[int(x2 + xl), int(y2 + yd)]
    g4 = g4 - GG[int(x2 + xl), int(y2 + yd)]
    b4 = b4 - BB[int(x2 + xl), int(y2 + yd)]

    error_r = r1*r1 + r2*r2 + r3*r3 + r4*r4
    error_g = g1*g1 + g2*g2 + g3*g3 + g4*g4
    error_b = b1*b1 + b2*b2 + b3*b3 + b4*b4

    error_r = math.sqrt(float(error_r)) * 100.0 / (255.0 * 2.0)
    error_g = math.sqrt(float(error_g)) * 100.0 / (255.0 * 2.0)
    error_b = math.sqrt(float(error_b)) * 100.0 / (255.0 * 2.0)

    return error_r, error_g, error_b


# ----------------------------
# Jeśli chcesz przetestować: uruchom bitmap_h('nazwa_pliku.jpg', 10, 10, 0.1, 3, False)
# ----------------------------
if __name__ == "__main__":
    # przykładowe wywołanie (podmień 'mp.JPG' na swój plik)
    import sys
    if len(sys.argv) >= 2:
        fname = sys.argv[1]
    else:
        fname = "ziemia.jpg"  # domyślnie, podmień na plik istniejący
    # elementsx i elementsy nie są używane w tym szkielecie — zostawione dla kompatybilności
    bitmap_h(fname, elementsxx=10, elementsyy=10, maxerror=1, max_refinement_level=10, color_edges_black=True)
