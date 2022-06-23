from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame, FrameObject
from movement_params.io.output import WindowOutput
from movement_params import CONFIG
from time import time
import cv2
import numpy as np
from numpy import linalg as LA
from movement_params.frame_processors.position_calculator import LocationConverter, PlaneFinder
from movement_params.CONFIG_STATIC import default_static as CFG


def middle(a: float, b: float, c: float):
    if a <= b and a <= c:
        if b <= c:
            mid = b
        else:
            mid = c
    else:
        if b <= a and b <= c:
            if a <= c:
                mid = a
            else:
                mid = c
        else:
            if a <= b:
                mid = a
            else:
                mid = b
    return mid


def collect_data(flag: int, data: float, id: int):
    if id == 30:
        if flag == 0:
            file = open('without.txt', 'a')
            file.write(str(data))
            file.write('\n')
            file.close()
        else:
            file = open('with.txt', 'a')
            file.write(str(data))
            file.write('\n')
            file.close()


class ParamsCalculator(FrameProcessor):
    def __init__(self, flag: int):
        # self.last_objects: list[FrameObject] = []
        self.__sec = time()
        self.flag = flag

    def process(self, frame: Frame) -> Frame:
        timebetweenframes = (time() - self.__sec)
        self.__sec = time()
        for obj in frame.objects:
            # for obj2 in self.last_objects:
            #     if obj.obj_id == obj2.obj_id:
            obj.wpcoord = obj.world_pos
            if len(obj.movement_params) > 2:
                obj2_params = obj.movement_params[-2]
                prevx, prevy = obj2_params.wpcoord
                curx, cury = obj.world_pos
                curspeedx, curspeedy = curx - prevx, cury - prevy
                obj.speedvec = (curspeedx, curspeedy)
                obj.speed = ((curspeedx ** 2 + curspeedy ** 2) ** 0.5) / timebetweenframes
                # collect_data(0, obj.speed, obj.obj_id)
                midspeed = middle(obj.movement_params[-3].speed, obj.movement_params[-2].speed,
                                  obj.movement_params[-1].speed)
                k = 0.5
                if abs(obj.speed - midspeed) > 20:
                    k = 0.3
                obj.speed = (obj.speed - midspeed) * k + midspeed
                # collect_data(1, obj.speed, obj.obj_id)
                k = 0.5
                # k - коэф, больше - точнее, меньше - плавнее
                if obj2_params.speed is not None:
                    # obj.acceleration = abs((obj.speed - obj2_params.speed) / timebetweenframes)
                    obj.acceleration = (((obj.speedvec[0] - obj2_params.speedvec[0])**2 + (obj.speedvec[1] - obj2_params.speedvec[1])**2)**0.5) / timebetweenframes
                    # collect_data(0, obj.acceleration, obj.obj_id)
                    midacc = middle(obj.movement_params[-3].acceleration, obj.movement_params[-2].acceleration,
                                    obj.movement_params[-1].acceleration)
                    if abs(obj.acceleration - midacc) > 10:
                        k = 0.3
                    obj.acceleration = (obj.acceleration - midacc) * k + midacc
                    # collect_data(1, obj.acceleration, obj.obj_id)
                    k = 0.5
        # self.last_objects = frame.objects
        # prediction
        ids = CFG.aruco_ids
        camera_matrix = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        world_matrix = CFG.world_matrix
        converter = LocationConverter(camera_matrix, world_matrix)
        plane_finder = PlaneFinder(ids)
        codes = plane_finder.get_aruco_codes(frame.image)
        plane = plane_finder.extract_plane(codes)
        plane = np.array(plane)
        converter.set_camera_matrix(plane)

        # camera_matrix = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        if self.flag == 1:
            sumx, sumy, sumx2, sumxy, sumy1, sumxy1 = 0, 0, 0, 0, 0, 0
            for obj in frame.objects:
                n = 3
                if len(obj.movement_params) > 4:
                    # last = [obj.movement_params[-5].coordinates,
                    #         obj.movement_params[-3].coordinates,
                    #         obj.movement_params[-1].coordinates]
                    last = [obj.movement_params[-5].wpcoord,
                            obj.movement_params[-3].wpcoord,
                            obj.movement_params[-1].wpcoord]
                    p = 0
                    for j in last:
                        sumx += p
                        sumx2 += p**2
                        sumy += j[0]
                        sumxy += p * j[0]
                        p += 1
                    a = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx ** 2 + 0.0001)
                    b = (sumy - a * sumx) / n
                    # x = int(a*(p + 10) + b)
                    x = a * (p + 10) + b
                    sumx, sumy, sumx2, sumxy, sumy1, sumxy1 = 0, 0, 0, 0, 0, 0
                    for i in last:
                        sumx += i[0]
                        sumy += i[1]
                        sumx2 += i[0]**2
                        sumxy += i[0]*i[1]
                    a = (n * sumxy - sumx * sumy)/(n * sumx2 - sumx**2 + 0.0001)
                    b = (sumy - a * sumx)/n
                    # y = int(a * x + b)
                    y = a * x + b
                    s = converter.world_to_camera((x, y))
                    obj.pred1 = (int(s[0]), int(s[1]))

        else:
            sumx, sumx2, sumx3, sumx4, sumy, sumyx, sumyx2 = 0, 0, 0, 0, 0, 0, 0
            for obj in frame.objects:
                n = 6
                if len(obj.movement_params) > 11:
                    bebra = [1, 2, 3, 4, 5, 6]
                    # last = [obj.movement_params[-12].coordinates,
                    #         obj.movement_params[-9].coordinates,
                    #         obj.movement_params[-6].coordinates,
                    #         obj.movement_params[-3].coordinates,
                    #         obj.movement_params[-2].coordinates,
                    #         obj.movement_params[-1].coordinates]
                    last = [obj.movement_params[-12].wpcoord,
                            obj.movement_params[-9].wpcoord,
                            obj.movement_params[-6].wpcoord,
                            obj.movement_params[-3].wpcoord,
                            obj.movement_params[-2].wpcoord,
                            obj.movement_params[-1].wpcoord]
                    for j in bebra:
                        sumx += j
                        sumx2 += j**2
                        sumx3 += j**3
                        sumx4 += j**4
                        sumyx += j * last[j-1][0]
                        sumyx2 += j * j * last[j-1][0]
                        sumy += last[j-1][0]
                    lm = [[sumx2, sumx, n], [sumx3, sumx2, sumx], [sumx4, sumx3, sumx2]]
                    rm = [sumy, sumyx, sumyx2]
                    x = LA.solve(lm, rm)
                    # x1 = int((n + 3) * x[0] + (n + 3) * x[1] + x[2])
                    # x2 = int((n + 6) * x[0] + (n + 6) * x[1] + x[2])
                    # x3 = int((n + 9) * x[0] + (n + 9) * x[1] + x[2])
                    x1 = (n + 3) * x[0] + (n + 3) * x[1] + x[2]
                    x2 = (n + 6) * x[0] + (n + 6) * x[1] + x[2]
                    x3 = (n + 9) * x[0] + (n + 9) * x[1] + x[2]
                    sumx, sumx2, sumx3, sumx4, sumy, sumyx, sumyx2 = 0, 0, 0, 0, 0, 0, 0

                    for i in last:
                        sumx += i[0]
                        sumx2 += i[0]**2
                        sumx3 += i[0]**3
                        sumx4 += i[0]**4
                        sumyx += i[0]*i[1]
                        sumyx2 += i[0]*i[1]*i[0]
                        sumy += i[1]
                    lm = [[sumx2, sumx, n], [sumx3, sumx2, sumx], [sumx4, sumx3, sumx2]]
                    rm = [sumy, sumyx, sumyx2]
                    x = LA.solve(lm, rm)
                    # y1 = int((x1 ** 2) * x[0] + x1 * x[1] + x[2])
                    # obj.pred1 = x1, y1
                    # y2 = int((x2 ** 2) * x[0] + x2 * x[1] + x[2])
                    # obj.pred2 = x2, y2
                    # y3 = int((x3 ** 2) * x[0] + x3 * x[1] + x[2])
                    # obj.pred3 = x3, y3
                    y1 = (x1 ** 2) * x[0] + x1 * x[1] + x[2]
                    s = converter.world_to_camera((x1, y1))
                    obj.pred1 = (int(s[0]), int(s[1]))
                    y2 = (x2 ** 2) * x[0] + x2 * x[1] + x[2]
                    s = converter.world_to_camera((x2, y2))
                    obj.pred1 = (int(s[0]), int(s[1]))
                    y3 = (x3 ** 2) * x[0] + x3 * x[1] + x[2]
                    s = converter.world_to_camera((x3, y3))
                    obj.pred1 = (int(s[0]), int(s[1]))
        return frame
