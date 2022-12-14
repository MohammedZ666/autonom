# importing the required packages
from threading import Thread
import cv2
import numpy as np
import time
import subprocess as sp
import torch
import os
from pytesseract import image_to_string
from tkinter import Canvas
import tkinter as tk


class GameStreamMidas:
    def __init__(self, queue, preview, test, limit):
        self.queue = queue
        self.thread = Thread(target=self.fetch_stream)
        self.preview = preview
        self.test = test
        self.limit = limit
        self.transform = None
        self.device = None
        self.midas = None

    def get_thread(self):
        return self.thread

    def bounding_box(self, img_gray):
        low, high = 191, 255
        ret, thresh = cv2.threshold(img_gray, low, high, cv2.THRESH_BINARY)
        kernel = np.array(cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3), (-1, -1)))
        img_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('abc', img_open)
        ret1, thresh1 = cv2.threshold(
            img_open, low, high, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(
            thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for i in range(len(contours)):
            if len(contours[i]) > 20:
                x, y, w, h = cv2.boundingRect(contours[i])
                cv2.rectangle(img_gray, (x, y), (x+w, y+h), (255, 255, 255), 2)
                print((x, y), (x+w, y+h))
        cv2.imshow('contours', img_gray)

    def get_region(self, start, end, img):
        # start is the top left point
        # end is the bottom right point
        # avg_list = np.empty((end[0]-start[0]+1))

        # for i in range(len(avg_list)):
        #     avg_list[i] = np.mean(img[:, i])
        # return (avg_list)
        img = img[start[1]: end[1], start[0]: end[0]]
        return img

    def initialize(self, model_type):
        # Load a MiDas model for depth estimation
        # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        if model_type == 0:
            model_type = "DPT_Large"
        if model_type == 1:
            # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
            model_type = "DPT_Hybrid"
        # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        else:
            model_type = "MiDaS_small"

        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # Move model to GPU if available
        # device = torch.device(
        #     "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

        # Load transforms to resize and normalize the img
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def infer(self, img):
        # Apply input transforms
        input_batch = self.transform(img).to(self.device)

        # Prediction and resize to original resolution
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        depth_map = cv2.normalize(depth_map, None, 0, 255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return depth_map

    def fetch_stream(self):

        width = 640
        height = int(width * 9/16)
        stream_fps = 2

        if self.test:
            command = 'ffmpeg -y -i video0.mp4 -pix_fmt bgr24 -vf scale=%s:-2 -vcodec rawvideo -an -sn -f image2pipe -' % width
        else:
            command = "ffmpeg -y -video_size 1920x1080 -framerate %s -f x11grab -i :0.0 -pix_fmt bgr24 -vf scale=%s:-2 -vcodec rawvideo -an -sn -f image2pipe -" % (
                stream_fps, width)

        pipe = sp.Popen(command.split(" "), stdout=sp.PIPE,
                        stderr=sp.PIPE, bufsize=-1)

        self.initialize(2)
        if self.preview:
            try:
                os.remove('depth_map.mp4')
                os.remove('video.mp4')
            except:
                pass

            # cv2.namedWindow("depth_map")
            # cv2.resizeWindow("depth_map", width, height)

            # cv2.namedWindow("game_stream")
            # cv2.resizeWindow("game_stream", width, height)
            writer_depth = cv2.VideoWriter('depth_map.mp4',
                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                           stream_fps, (width, height))
            writer_image = cv2.VideoWriter('video.mp4',
                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                           stream_fps, (width, height))

        # x = int((width * 0.5))
        # y = int((height * 0.5))
        # h_factor = 1/128
        # w_factor = 1/128
        # h_off = int(height * 1/12) * -1

        x = int((width * 0.5))
        y = int((height * 0.5))
        gap = int(width * 1/16)
        h_factor = 1/32
        w_factor = 1/32
        h_off = int(height * 1/24) * -1

        # x = int((width * 0.5))
        # y = int((height * 0.5))
        # gap = int(width * 1/16)
        # h_factor = 1/24
        # w_factor = 1/16
        # h_off = int(height * 1/32) * -1

        # x = int((width * 0.5))
        # y = int((height * 0.5))
        # gap = int(width * 1/16)
        # h_factor = 1/64
        # w_factor = 1/32
        # h_off = int(height * 1/12) * -1

        x0, y0 = (int(x - width * w_factor),
                  int(y - height * h_factor) - h_off)
        x1, y1 = (int(x + width * w_factor),
                  int(y + height * h_factor) - h_off)

        x2, y2 = (int(gap + x + width * w_factor),
                  int(y - height * h_factor) - h_off)

        x3, y3 = (int(gap + x + 3 * width * w_factor),
                  int(y + height * h_factor) - h_off)

        x4, y4 = (int(x - 3 * width * w_factor - gap),
                  int(y - height * h_factor) - h_off)

        x5, y5 = (int(x - width * w_factor - gap),
                  int(y + height * h_factor) - h_off)

        points = np.array(

            [
                # cv2.imshow("depth_map", depth_map)
                [[x0, y0], [x1, y1]],

                [[x2, y2], [x3, y3]],
                [[x4, y4], [x5, y5]],


                [[x0, y0 + gap], [x1, y1 + gap]],
                [[x2, y2 + gap], [x3, y3 + gap]],
                [[x4, y4 + gap], [x5, y5 + gap]],

                [[x0, y0 - gap], [x1, y1 - gap]],
                [[x2, y2 - gap], [x3, y3 - gap]],
                [[x4, y4 - gap], [x5, y5 - gap]]
            ],

        )
        color = (0,  0, 0)
        thickness = 5
        all_white = (x0-x1) * (y0-y1) * 255
        # xs0, ys0 = (width - int(width*1/8),
        #             y - 3 * int(height * 1/32))
        # xs1, ys1 = (width - int(width * 1/20), y - int(height * 1/24))
        while True:
            start = time.time()

            img = np.frombuffer(
                buffer=pipe.stdout.read(width*height*3), dtype='uint8')
            img = img.reshape((height, width, 3))
            depth_map = self.infer(img)
            depth_map = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # val = 0
            # for p in points:
            #     # cv2.rectangle(depth_map, p[0], p[1], color, thickness)
            #     cv2.rectangle(img, p[0], p[1], color, thickness)
            #     val = (
            #         val + self.get_region(p[0], p[1], depth_map))/2

            _, depth_map = cv2.threshold(
                depth_map, 40, 255, cv2.THRESH_BINARY)

            region = self.get_region(
                points[0][0], points[0][1], depth_map)

            val = (region.flatten().sum())*100/all_white
            h_region = region[0:int((x0-x1)/2), 0:y0-y1]

            if val < self.limit and h_region.sum() > 0:
                val = self.limit

            self.queue.put(val)

            # speed_region = img[ys0:ys1,  xs0:xs1]
            # speed_region = cv2.cvtColor(speed_region, cv2.COLOR_BGR2HSV)
            # lower_region = np.array([98, 96, 112], np.uint8)
            # upper_region = np.array([179, 168, 255], np.uint8)
            # digits = cv2.inRange(speed_region, lower_region, upper_region)
            # kernel = np.ones((1, 1), "uint8")
            # red = cv2.morphologyEx(digits, cv2.MORPH_OPEN, kernel)
            # red = cv2.dilate(red, kernel, iterations=1)
            # speed_region = 255 - \
            #     cv2.bitwise_and(speed_region, speed_region, mask=red)
            # speed_region[speed_region < 255] = 0
            # speed = image_to_string(
            #     speed_region, lang='eng',
            #     config='-c tessedit_char_whitelist=0123456789 --psm 7')
            # print(speed.replace('?', ''))

            val = "STOP: %s" % (
                int(val)) if val > self.limit else "GO: %s" % (
                int(val))

            if self.preview:
                # cv2.imshow("speed", speed_region)
                cv2.imshow("region", region)
                # cv2.imshow("h_region", h_region)

                cv2.rectangle(depth_map, points[0][0],
                              points[0][1], color, 1)
                cv2.rectangle(img, points[0][0],
                              points[0][1],  color, 1)
                # cv2.rectangle(img, (xs0, ys0),
                #               (xs1, ys1),  color, thickness)

                # cv2.circle(img, (x, y), 0, color, thickness)
                # cv2.circle(depth_map, (x, y), 0, color, thickness)

                end = time.time()
                totalTime = end - start
                current_fps = 1 / totalTime

                cv2.putText(img, f'FPS: {int(current_fps)}', (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(img, f'VAL: {str(val)}', (5, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(depth_map, f'FPS: {int(current_fps)}', (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.putText(depth_map, f'VAL: {str(val)}', (5, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
                # depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

                cv2.imshow("game_stream", img)
                # cv2.imshow("depth_map", depth_map)
                # writer_depth.write(depth_map)
                # writer_image.write(img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pipe.stdout.flush()
                    pipe.kill()
                    break

            pipe.stdout.flush()

            if self.test:
                time.sleep(0.3)
        if self.preview:
            cv2.destroyAllWindows()
