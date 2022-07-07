#!/usr/bin/env python
from matplotlib import pyplot as plt
from PIL import Image
from glob import glob
import numpy as np
import ast
import argparse
import re
import os
import copy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from shapely.geometry import Polygon
class draw_bb:
    def __init__(self, args):
        self.args = args

    def read_image(self,img_path):
        img = Image.open(img_path)
        return np.array(img)

    def get_pedestrian_box(self,txt_path):
        data = open(txt_path).readlines()
        bbox = []
    #     labels = []
        for d in data:
            d = d.rstrip()
            d = np.array(ast.literal_eval(d))
            
            x1, x2 = sorted([max(d[:, 0]), min(d[:, 0])])
            y1, y2 = sorted([max(d[:, 1]), min(d[:, 1])])

            bbox.append(([x1, y1, x2, y2]))
    #         labels.append(-1)
        return bbox#, labels

    def get_vehicle_box(self,txt_path):
        data = open(txt_path).readlines()
        
        boxs = []
        labels = []
        
        for d in data:
            d, label = d.rstrip().split('\t')
            d = np.array(ast.literal_eval(d))
            
            x1, x2 = sorted([max(d[:, 0]), min(d[:, 0])])
            y1, y2 = sorted([max(d[:, 1]), min(d[:, 1])])

            boxs.append(([x1, y1, x2, y2]))
            labels.append(label)
        return boxs, labels




    def remove_small_box(self, boxes, labels, size_threshold, x_threshold = 0, y_threshold = 0):
        if len(boxes) == 0:
            return boxes, labels
        
        Mask = [False for _ in range(len(boxes))]
        boxes = np.array(boxes)
        labels = np.array(labels)
        for i, box in enumerate(boxes):
            xmin, xmax = min(box[0], box[2]), max(box[0], box[2])
            ymin, ymax = min(box[1], box[3]), max(box[1], box[3])
            
            x_size = xmax - xmin
            y_size = ymax-ymin
            
            if x_size > x_threshold and y_size > y_threshold and x_size * y_size > size_threshold:
                Mask[i] = True

        return boxes[Mask], labels[Mask]

    def remove_unseen_box(self, boxes, labels, segmentation, label, threshold, margin = 0):
        if len(boxes) == 0:
            return boxes, labels

        Mask = [False for _ in range(len(boxes))]
        boxes = np.array(boxes)
        labels = np.array(labels)

        r,g,b = label
        lim_y, lim_x = segmentation.shape[:2]
        
        for i, box in enumerate(boxes):
            xmin, xmax = max(min(box[0], box[2], lim_x), 0), min(max(box[0], box[2], 0), lim_x)
            ymin, ymax = max(min(box[1], box[3], lim_y), 0), min(max(box[1], box[3], 0), lim_y)
            if xmin == xmax or ymin == ymax:
                continue

            in_box_img = copy.deepcopy(segmentation[ymin:ymax + 1, xmin:xmax + 1, :])

            in_box_img[:, :, 0] = (r - margin <= in_box_img[:, :, 0]) * (in_box_img[:, :, 0] <= r + margin)
            in_box_img[:, :, 1] = (g - margin <= in_box_img[:, :, 1]) * (in_box_img[:, :, 1] <= g + margin)
            in_box_img[:, :, 2] = (b - margin <= in_box_img[:, :, 2]) * (in_box_img[:, :, 2] <= b + margin)

            correct = np.sum(in_box_img[:, :, 0] * in_box_img[:, :, 1] * in_box_img[:, :, 2])
            if correct / ((xmax+1 - xmin) * (ymax+1 - ymin)) > threshold:
                Mask[i] = True
        return boxes[Mask], labels[Mask]
    def clamp(self, box, xmax, ymax):
        return max(0, min(box[0], xmax)), \
                max(0, min(box[1], ymax)), \
                max(0, min(box[2], xmax)), \
                max(0, min(box[3], ymax))
        

    def remove_strange_box(self, boxes, img_shape): #경계를 넘어가는 box제거
        boxes = np.array(boxes)
        y_max, x_max, _ = img_shape
        
        for i in range(len(boxes)):
            boxes[i] = self.clamp(boxes[i], x_max, y_max)
        
        return boxes

    
    polygon = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
    other_polygon = Polygon([(1, 1), (4, 1), (4, 3.5), (1, 3.5)])
    intersection = polygon.intersection(other_polygon)
    intersection.area

    def remove_overlap_box(self, boxs, labels, threshold):
        if len(boxs) == 0:
            return boxs, labels
        
        boxs = np.array(boxs)
        labels = np.array(labels)
        x1,x2 = boxs[:, 0], boxs[:, 2]
        y1,y2 = boxs[:, 1], boxs[:, 3]
        area = ((x2-x1) * (y2-y1)).argsort() # 우선 작은것부터 겹치면 지우자. 그럼 자연히 큰거만 남게 된다.
        
        Mask = [True for _ in range(len(boxs))]

        polygons = [Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]) for box in boxs]

        for i in area:
            overlap_ratio = [-1]
            for j in range(len(boxs)):
                if i!= j and Mask[j]:
                    intersection = polygons[i].intersection(polygons[j]).area
                    overlap_ratio.append(intersection/polygons[i].area)
                    
            if max(overlap_ratio) > threshold:
                Mask[i] = False
        
        
        return boxs[Mask], labels[Mask]

    # boxs : box정보 + label정보
    # bboxs: box정보만
    # import torch
    # 사람 : (220, 020, 060)
    # 차, 오토바이 : (000, 000, 142)
    # 사람 : (220, 020, 060)

    
    def make_filtered_bb(self):
        image_path = sorted(glob(f"custom_data/{self.args.map}/*.png"))
        segimage_path = sorted(glob(f"SegmentationImage/{self.args.map}/*.png"))
        ped_box_path = sorted(glob(f"PedestrianBBox/{self.args.map}/b*"))
        veh_box_path = sorted(glob(f"VehicleBBox/{self.args.map}/b*"))

        person = (220, 20, 60)
        car = (0, 0, 142)

        color_map = ['red', 'blue']
        ped_color = 'black'

        for i in range(len(veh_box_path)):
            # i = 3
            # print(i)
            regex = re.compile(r'\d+')
            number = int(regex.findall(os.path.basename(veh_box_path[i]))[0])
            plt.figure(figsize = (54/2, 96/2))
            img = self.read_image(segimage_path[i])

            boxs = np.array(self.get_pedestrian_box(ped_box_path[i]))
            labels = np.array([1 for _ in range(len(boxs))])
            boxs = self.remove_strange_box(boxs, img.shape)
            boxs, labels = self.remove_small_box(boxs, labels, size_threshold = 30)
            boxs, labels = self.remove_unseen_box(boxs, labels, img, person, threshold = 0.10, margin = 10)
            boxs, labels = self.remove_overlap_box(boxs, labels = [1 for _ in range(len(boxs))], threshold= 0.8)
            f = open(f"PedestrianBBox/{self.args.map}/filtered_bbox"+str(number), 'w')
            for box in boxs:
                # plt.plot([box[0], box[0],box[2],box[2],box[0]], [box[1], box[3],box[3],box[1],box[1]], color = ped_color)
                f.write("4 " + " ".join(map(str, box)) + "\n")
            
            boxs, labels = self.get_vehicle_box(veh_box_path[i])
            labels= np.array(labels)
            boxs = np.array(boxs)
            boxs = self.remove_strange_box(boxs, img.shape)
            boxs, labels = self.remove_small_box(boxs,labels, size_threshold = 150)
            boxs, labels = self.remove_unseen_box(boxs,labels, img, car, threshold = 0.10, margin = 10)
            boxs, labels = self.remove_overlap_box(boxs, labels, 0.9)
            f = open(f"VehicleBBox/{self.args.map}/filtered_bbox"+str(number), 'w')
            for ii, box in enumerate(boxs):
                # if labels[ii] == '2':
                #     plt.plot([box[0], box[0],box[2],box[2],box[0]], [box[1], box[3],box[3],box[1],box[1]], linewidth = 8, color = color_map[0])
                # else:
                #     plt.plot([box[0], box[0],box[2],box[2],box[0]], [box[1], box[3],box[3],box[1],box[1]], linewidth = 8, color = color_map[1])
                f.write(str(labels[ii]) + " " + " ".join(map(str, box)) + "\n")
            # img = read_image(image_path[i])
            # plt.imshow(img)

            # plt.show()
            # break

# if __name__ == '__main__':
#     main()