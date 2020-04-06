import cv2
import numpy as np
import os

class ExtractTable:
    def __init__(self, filepath, debug=False):
        self.filepath   = filepath
        self.debug      = debug

    def sort_contours(self, cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
        
        return (cnts, boundingBoxes)

    def isolate_lines(self, src, structuring_element):
        cv2.erode(src, structuring_element, src, (-1, -1)) # makes white spots smaller
        cv2.dilate(src, structuring_element, src, (-1, -1)) # makes white spots bigger

    
    def verify_table(self, contour, intersections):
        MIN_TABLE_AREA = 50             # min table area to be considered a table
        EPSILON = 3                     # epsilon value for contour approximation
        area    = cv2.contourArea(contour)

        if (area < MIN_TABLE_AREA):
            return (None, None)

        # approxPolyDP approximates a polygonal curve within the specified precision
        curve   = cv2.approxPolyDP(contour, EPSILON, True)

        # boundingRect calculates the bounding rectangle of a point set (eg. a curve)
        rect    = cv2.boundingRect(curve) # format of each rect: x, y, w, h

        # Finds the number of joints in each region of interest (ROI)
        # Format is in row-column order (as finding the ROI involves numpy arrays)
        # format: image_mat[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w]
        possible_table_region       = intersections[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        (possible_table_joints, _)  = cv2.findContours(possible_table_region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Determines the number of table joints in the image
        # If less than 5 table joints, then the image
        # is likely not a table
        if len(possible_table_joints) < 5:
            return (None, None)

        return rect, possible_table_joints

    def getTablesV1(self):
        rects           = []
        if os.path.exists(self.filepath):
            src_img                 = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
            gray                    = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            MAX_THRESHOLD_VALUE     = 255
            BLOCK_SIZE              = 15
            THRESHOLD_CONSTANT      = 0
            SCALE                   = 15

            # Filter image
            filtered                = cv2.adaptiveThreshold(~gray, MAX_THRESHOLD_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, THRESHOLD_CONSTANT)

            horizontal              = filtered.copy()
            horizontal_size         = int(horizontal.shape[1] / SCALE)
            horizontal_structure    = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
            self.isolate_lines(horizontal, horizontal_structure)

            vertical                = filtered.copy()
            vertical_size           = int(vertical.shape[0] / SCALE)
            vertical_structure      = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
            self.isolate_lines(vertical, vertical_structure)

            mask                    = horizontal + vertical
            contours                = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours                = contours[0] if len(contours) == 2 else contours[1]
            intersections           = cv2.bitwise_and(horizontal, vertical)

            if self.debug:
                print('V1: total contours found %d' % ( len(contours) ))

            tables                  = []
            for i in range(len(contours)):
                (rect, table_joints) = self.verify_table(contours[i], intersections)
                if rect == None or table_joints == None:
                    continue
                rects.append(rect)
        if self.debug:
            print('V1: found %d tables' % (len(rects)))
        return rects

    def getTables(self):
        rects           = []
        if os.path.exists(self.filepath):
            src_img     = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
            blur_img    = cv2.pyrMeanShiftFiltering(src_img, 11, 21)
            gray_img    = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
            bw_img      = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
            
            contours    = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours    = contours[0] if len(contours) == 2 else contours[1]
            if self.debug:
                print('V0: total contours found %d' % ( len(contours) ))

            for c in contours:
                peri    = cv2.arcLength(c, True)
                approx  = cv2.approxPolyDP(c, 0.015 * peri, True)
                if len(approx) == 4:
                    x,y,w,h = cv2.boundingRect(approx)
                    rects.append((x,y,w,h))

        if self.debug:
            print('V1: found %d tables' % (len(rects)))
        return rects

    def getTableImage(self, rect):
        EXTRA_PIXEL = 20
        img      = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
        x,y,w,h  = rect
        crop     = img[y-EXTRA_PIXEL:y-EXTRA_PIXEL+h+2*EXTRA_PIXEL, x-EXTRA_PIXEL:x-EXTRA_PIXEL+w+2*EXTRA_PIXEL]
        return crop

    def getTableRects(self, rect):
        SCALE               = 30
        src_img             = self.getTableImage(rect)
        gray_img            = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        thresh, bw_img      = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bw_img              = 255-bw_img
        
        vertical_size       = int(src_img.shape[0] / SCALE)
        ver_kernel          = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

        horizontal_size     = int(src_img.shape[1] / SCALE)
        hor_kernel          = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        kernel              = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Use vertical kernel to detect and save the vertical lines
        vertical_img        = cv2.erode(bw_img, ver_kernel, iterations=3)
        vertical_img        = cv2.dilate(vertical_img, ver_kernel, iterations=3)
        
        # Use horizontal kernel to detect and save the horizontal lines
        horizontal_img      = cv2.erode(bw_img, hor_kernel, iterations=3)
        horizontal_img      = cv2.dilate(horizontal_img, hor_kernel, iterations=3)
        
        # Combine horizontal and vertical lines in a new third image, with both having same weight.
        img_vh              = cv2.addWeighted(vertical_img, 0.5, horizontal_img, 0.5, 0.0)
        # Eroding and thesholding the image
        img_vh              = cv2.erode(~img_vh, kernel, iterations=2)
        thresh, img_vh      = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bitxor              = cv2.bitwise_xor(gray_img, img_vh)
        bitnot              = cv2.bitwise_not(bitxor)
        
        # Detect contours for following box detection
        contours            = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours            = contours[0] if len(contours) == 2 else contours[1]
        if self.debug:
            print('total contours found %d' % ( len(contours) ))

        # Sort all the contours by top to bottom.
        contours, boundingBoxes = self.sort_contours(contours, method="top-to-bottom")
        rects = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (abs(src_img.shape[0] - h) < 15) or (abs(src_img.shape[1] - w) < 15):
                continue
            if h < 10:
                continue    
            rects.append([x,y,w,h])
            
        return rects