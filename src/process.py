import cv2
import numpy as np
from .table import Table
from . import utils

def process_tables(filepath):
    img                     = cv2.imread(filepath)
    gray                    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    MAX_THRESHOLD_VALUE     = 255
    BLOCK_SIZE              = 15
    THRESHOLD_CONSTANT      = 0
    SCALE                   = 15

    # Filter image
    filtered                = cv2.adaptiveThreshold(~gray, MAX_THRESHOLD_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)

    horizontal              = filtered.copy()
    horizontal_size         = int(horizontal.shape[1] / SCALE)
    horizontal_structure    = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    utils.isolate_lines(horizontal, horizontal_structure)

    vertical                = filtered.copy()
    vertical_size           = int(vertical.shape[0] / SCALE)
    vertical_structure      = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    utils.isolate_lines(vertical, vertical_structure)

    mask                    = horizontal + vertical
    contours                = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours                = contours[0] if len(contours) == 2 else contours[1]
    intersections           = cv2.bitwise_and(horizontal, vertical)

    tables                  = []
    for i in range(len(contours)):
        (rect, table_joints) = utils.verify_table(contours[i], intersections)
        if rect == None or table_joints == None:
            continue

        table           = Table(rect[0], rect[1], rect[2], rect[3])
        joint_coords    = []

        for i in range(len(table_joints)):
            joint_coords.append(table_joints[i][0][0])
        joint_coords    = np.asarray(joint_coords)
        sorted_indices  = np.lexsort((joint_coords[:, 0], joint_coords[:, 1]))
        joint_coords    = joint_coords[sorted_indices]
        table.set_joints(joint_coords)
        tables.append(table)
    return tables

def process_lines(filepath, length=50):
    img                     = cv2.imread(filepath)
    gray                    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    MAX_THRESHOLD_VALUE     = 255
    BLOCK_SIZE              = 15
    THRESHOLD_CONSTANT      = 0
    SCALE                   = 15

    # Filter image
    filtered                = cv2.adaptiveThreshold(~gray, MAX_THRESHOLD_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)
    horizontal              = filtered.copy()
    horizontal_size         = int(horizontal.shape[1] / SCALE)
    horizontal_structure    = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    utils.isolate_lines(horizontal, horizontal_structure)

    contours                = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours                = contours[0] if len(contours) == 2 else contours[1]

    # filtered_contours       = []
    lines                   = []

    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > length:
            # filtered_contours.append(contour)
            lines.append((x,y,w,h))
    return lines

def detect_tables_and_lines(filepath):
    ts = process_tables(filepath)
    ls = process_lines(filepath)

    table_coordinates = []
    lines_coordinates = []
    for t in ts:
        table_coordinates.append((t.y, t.y+t.h))
    
    for l in ls:
        within_table = False

        for index, table_coordinate in enumerate(table_coordinates):
            print('Rect Y1: %d <= line y1: %d <= Rect Y2: %d' % (table_coordinate[0], l[1], table_coordinate[1]))
            if l[1] >= table_coordinate[0] and l[1] <= table_coordinate[1]:
                print('line x1: %d, y: %d, is within table index %d' % (l[0], l[1], index))
                within_table = True
                break
        if within_table == False:
            lines_coordinates.append(l)

    tables = []
    lines  = []
    for t in ts:
        tables.append({'x': t.x, 'y': t.y, 'w': t.w, 'h': t.h})
    for l in lines_coordinates:
        lines.append({'x': l[0], 'y': l[1], 'w': l[2], 'h': l[3]})
    return tables, lines