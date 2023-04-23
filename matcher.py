import math
import os

import numpy as np
from cv2 import cv2
from cv2 import cv2 as cv


class Matcher(object):
    MAX_ANG_ERROR = 5
    MIN_ANG = 10
    MAX_RADIUS = 10
    MIN_RADIUS = 3
    MIN_PIXEL_DISTANCE = 30
    BW_THRESH_LOWER = 200
    BW_THRESH_UPPER = 255
    ACCEPTABLE_RATIO_ERROR = 0.07

    def __init__(self, src_fname, dest_fname):
        self.src_fname = src_fname
        self.dest_fname = dest_fname

        self.saved_circles = {}
        self.data = {}

    @staticmethod
    def angle_between(a, b, c):
        a = np.array([a[0], a[1]])
        b = np.array([b[0], b[1]])
        c = np.array([c[0], c[1]])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)


    @staticmethod
    def saved_circles_to_kd_tree(saved_circles: dict, to_ignore=[]):
        from sklearn.neighbors import KDTree
        ret = {}
        for f in saved_circles:
            data_to_append = []
            for x in saved_circles[f]:
                entry = x[:2]
                found = False
                for ignored in to_ignore:
                    if all(ignored[:2] == entry):
                        found = True
                        break
                if not found:
                    data_to_append.append(entry)
            tree = KDTree(data_to_append, leaf_size=2)
            ret[f] = tree
        return ret

    def detect_circles(self):
        try:
            dirname = ("results-{}-{}".format(self.src_fname, self.dest_fname))
            os.makedirs(dirname)
            os.chdir(dirname)
        except:
            pass

        self.saved_circles = {}
        self.data = {}

        for f in [self.src_fname, self.dest_fname]:
            self.data[f] = {}

            img = cv.imread("../imgs/{}".format(f), cv.IMREAD_GRAYSCALE)
            print("../imgs/{}".format(f))
            (thresh, im_bw) = cv.threshold(img, self.BW_THRESH_LOWER, self.BW_THRESH_UPPER, cv.THRESH_BINARY)

            output = img.copy()
            output = cv2.bitwise_not(output)

            circles = cv2.HoughCircles(im_bw, cv2.HOUGH_GRADIENT, 1, self.MIN_PIXEL_DISTANCE, param1=5, param2=8,
                                       maxRadius=self.MAX_RADIUS, minRadius=self.MIN_RADIUS)
            # ensure at least some circles were found
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                self.saved_circles[f] = np.copy(circles)
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(output, (x, y), r * 3, (123, 255, 123), 4)

            cv.imwrite("bw-{}".format(f), im_bw)
            cv.imwrite("all-detected-{}".format(f), np.hstack([img, output]))

            if circles is  None:
                return

            for c1 in circles:
                for c2 in circles:
                    for c3 in circles:
                        if all(c1 == c3) or all(c1 == c2) or all(c2 == c3):
                            continue

                        rounded_angle = round(Matcher.angle_between(c1, c2, c3))
                        if rounded_angle >= self.MIN_ANG:
                            if rounded_angle not in self.data[f]:
                                self.data[f][rounded_angle] = []
                            self.data[f][rounded_angle].append([np.copy(c1), np.copy(c2), np.copy(c3)])

        self.kd_save_circles = Matcher.saved_circles_to_kd_tree(self.saved_circles)


    def find_similarities(self):
        best_error = math.inf
        count = 0

        for ang_img1 in self.data[self.src_fname]:
            for res_src in self.data[self.src_fname][ang_img1]:
                for ang in self.data[self.dest_fname]:
                    if abs(ang_img1 - ang) <= self.MAX_ANG_ERROR:
                        for res_dest in self.data[self.dest_fname][ang]:

                            # Get 3 points for this angle
                            src_c1 = res_src[0]
                            src_c2 = res_src[1]
                            src_c3 = res_src[2]

                            # Get vector length
                            delta_src_c2c1 = src_c2 - src_c1
                            delta_src_c2c3 = src_c2 - src_c3

                            # Get 3 points for this angle from second image
                            dest_c1 = res_dest[0]
                            dest_c2 = res_dest[1]
                            dest_c3 = res_dest[2]

                            delta_dest_c2c1 = dest_c2 - dest_c1
                            delta_dest_c2c3 = dest_c2 - dest_c3

                            ratio_c2c1 = np.linalg.norm(delta_src_c2c1) / np.linalg.norm(delta_dest_c2c1)
                            ratio_c2c3 = np.linalg.norm(delta_dest_c2c3 * ratio_c2c1) / np.linalg.norm(delta_src_c2c3)

                            # How warped is the picture?
                            ratio_diff = abs(ratio_c2c3 - 1)
                            if ratio_diff >= self.ACCEPTABLE_RATIO_ERROR:
                                continue

                            # kd_tree.valid_metrics
                            dist, ind = self.kd_save_circles[self.src_fname].query([src_c2[:2]], k=3)
                            dist2, ind2 = self.kd_save_circles[self.dest_fname].query([dest_c2[:2]], k=3)

                            neighbour_ratios = abs((dist2 * ratio_c2c1 / dist) - 1)
                            neighbours_violated_ratio_threshold = neighbour_ratios > self.ACCEPTABLE_RATIO_ERROR

                            should_cont = False
                            for r in neighbours_violated_ratio_threshold:
                                if any(r):
                                    should_cont = True
                                    break

                            if should_cont:
                                break


                            # Now find by neighbours from the other vertex of triangle
                            distc3, indc3 = self.kd_save_circles[self.src_fname].query([src_c1[:2]], k=2)
                            distc3_2, indc3_2 = self.kd_save_circles[self.dest_fname].query([dest_c1[:2]], k=2)

                            neightbour_ratios2 = abs((distc3_2 * ratio_c2c1 / distc3) - 1)
                            neighbous_violated_ratio_threshold2 = neightbour_ratios2 > self.ACCEPTABLE_RATIO_ERROR

                            should_cont = False
                            for r in neighbous_violated_ratio_threshold2:
                                if any(r):
                                    should_cont = True
                                    break
                            if should_cont:
                                continue


                            best_point = [res_src, res_dest]

                            # OUTPUT
                            img1 = cv.imread("../imgs/{}".format(self.src_fname), cv.IMREAD_GRAYSCALE)
                            img2 = cv.imread("../imgs/{}".format(self.dest_fname), cv.IMREAD_GRAYSCALE)

                            # img2 = cv2.resize(img2, (0,0), fx=ratio_c2c1, fy=ratio_c2c1)
                            # # Scale circles by found ratio
                            # best_point[1] = np.matmul(best_point[1], np.array([[ratio_c2c1, 0,0],
                            #                                                    [0, ratio_c2c1, 0],
                            #                                                    [0 , 0, 1]])).astype(int)

                            # ang_src_c2_x_axis = angle_between([0, 0], src_c2, [999999, 0])
                            # ang_dest_c2_x_axis = angle_between([0, 0], dest_c2, [999999, 0])

                        #     M = cv2.getRotationMatrix2D((0, 0),abs(ang_dest_c2_x_axis - ang_src_c2_x_axis), 1)
                        #     img2 = cv2.warpAffine(img2,M,(img2.shape[1],img2.shape[0]))


                            cv2.circle(img1, (best_point[0][0][0], best_point[0][0][1]), 20 , (255, 255, 255), 4)
                            cv2.circle(img1, (best_point[0][1][0], best_point[0][1][1]), 20 , (123, 255, 123), 4)
                            cv2.circle(img1, (best_point[0][2][0], best_point[0][2][1]), 20 , (255, 255, 255), 4)

                            cv2.circle(img2, (best_point[1][0][0], best_point[1][0][1]), 20 , (255, 255, 255), 4)
                            cv2.circle(img2, (best_point[1][1][0], best_point[1][1][1]), 20 , (123, 255, 123), 4)
                            cv2.circle(img2, (best_point[1][2][0], best_point[1][2][1]), 20 , (255, 255, 255), 4)

                            cv.imwrite("res-{}-{}-{}".format(count,best_point[0], self.src_fname), img1)
                            cv.imwrite("res-{}-{}-{}".format(count, best_point[1] ,self.dest_fname), img2)

                            count+=1