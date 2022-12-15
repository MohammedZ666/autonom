# importing the required packages
from threading import Thread
import cv2
import numpy as np
import time
import subprocess as sp
import torch
import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from monodepth2 import networks
from monodepth2.utils import download_model_if_doesnt_exist


class GameStream:
    def __init__(self, queue, preview):
        self.queue = queue
        self.thread = Thread(target=self.fetch_stream)
        self.preview = preview
        self.feed_height = None
        self.feed_width = None
        self.encoder = None
        self.depth_decoder = None

    def get_thread(self):
        return self.thread

    def get_region(self, start, end, img):
        # start is the top left point
        # end is the bottom right point
        avg_list = np.empty((end[0]-start[0]+1))

        for i in range(len(avg_list)):
            avg_list[i] = np.mean(img[:, i])
        return (int(avg_list.mean()))
        # print(np.diff(avg_list))

    def intialize_model(self):
        ROOT = ''
        model_name = "mono_640x192"
        download_model_if_doesnt_exist(model_name)
        encoder_path = os.path.join(ROOT, "models", model_name, "encoder.pth")
        depth_decoder_path = os.path.join(
            ROOT,  "models", model_name, "depth.pth")

        # LOADING PRETRAINED MODEL
        encoder = networks.ResnetEncoder(18, False)
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        depth_decoder.load_state_dict(loaded_dict)

        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        print(encoder.eval(), depth_decoder.eval())
        self.encoder = encoder
        self.depth_decoder = depth_decoder

    def infer(self, input_image, original_width, original_height):

        input_image_resized = cv2.resize(input_image, dsize=(
            self.feed_width, self.feed_height), interpolation=cv2.INTER_LANCZOS4)

        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(input_image_pytorch)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp,
                                                       (original_height, original_width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()

        return disp_resized_np

    def fetch_stream(self):
        width = 640
        height = int(width * 9/16)

        # command = "ffmpeg -y -video_size 1920x1080 -framerate 2 -f x11grab -i :0.0 -pix_fmt bgr24 -vf fps=2,scale=%s:-2 -vcodec rawvideo -an -sn -f image2pipe -" % width
        command = "ffmpeg -y -i output1.mkv -pix_fmt bgr24 -vf fps=2,scale=%s:-2 -vcodec rawvideo -an -sn -f image2pipe -" % width
        pipe = sp.Popen(command.split(" "), stdout=sp.PIPE,
                        stderr=sp.PIPE, bufsize=-1)

        if self.preview:
            cv2.namedWindow("depth_map")
            cv2.resizeWindow("depth_map", width, height)

            cv2.namedWindow("game_stream")
            cv2.resizeWindow("game_stream", width, height)
            writer = cv2.VideoWriter('depth_map.mp4',
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     15, (width, height))

        x = int((width * 0.5))
        y = int((height * 0.5))
        h_factor = 1/128
        w_factor = 1/128
        h_off = int(height * 1/12) * -1

        # x = int((width * 0.5))
        # y = int((height * 0.5))
        gap = int(width * 1/16)
        # h_factor = 1/64
        # w_factor = 1/64
        # h_off = int(height * 1/16)* -1
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
        self.intialize_model()
        while True:
            start = time.time()
            img = np.frombuffer(
                buffer=pipe.stdout.read(width*height*3), dtype='uint8')
            img = img.reshape((height, width, 3))
            depth_map = self.infer(img, width, height)
            depth_map = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            val = self.get_region(
                points[0][0], points[0][1], depth_map)
            self.queue.put(0)

            if self.preview:
                cv2.rectangle(depth_map, points[0][0],
                              points[0][1], color, thickness)
                cv2.rectangle(img, points[0][0],
                              points[0][1],  color, thickness)

                cv2.circle(img, (x, y), 0, color, thickness)
                cv2.circle(depth_map, (x, y), 0, color, thickness)

                end = time.time()
                totalTime = end - start
                fps = 1 / totalTime

                cv2.putText(depth_map, f'FPS: {int(fps)}', (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(depth_map, f'VAL: {str(val)}', (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

                cv2.imshow("game_stream", img)
                cv2.imshow("depth_map", depth_map)
                # writer.write(depth_map)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pipe.kill()
                    break

            pipe.stdout.flush()

        if self.preview:
            cv2.destroyAllWindows()