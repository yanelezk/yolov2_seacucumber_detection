import argparse
import logging as log
from statistics import mean
import torch
from torchvision.transforms.functional import to_tensor
import copy
import os
from vedanet import data as vn_data
from vedanet import models
from vedanet import engine
from utils.test import voc_wrapper
from pprint import pformat
import brambox.boxes as bbb
import vedanet as vn
from torch.autograd import Variable
from utils.envs import initEnv
import cv2
import numpy as np
from vedanet.engine._voc_test import CustomDataset
from PIL import Image
from utils.test.fast_rcnn.nms_wrapper import nms, soft_nms
from PIL import Image, ImageOps
import sys
sys.path.insert(0, '.')
from df import Ui_Form
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork
import datetime
from PyQt5.QtCore import pyqtSlot,pyqtSignal,QByteArray,QDataStream,qUncompress
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QApplication
from df import Ui_Form

class VOCTest(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(VOCTest, self).__init__()
        # 界面初始化
        self.setupUi(self)
        parser = argparse.ArgumentParser(description='OneDet: an one stage framework based on PyTorch')
        parser.add_argument('--model_name', dest='model_name', help='model name', default='Yolov2')
        args = parser.parse_args()
        train_flag = 2
        config = initEnv(train_flag=train_flag, model_name=args.model_name)
        # init env
        hyper_params = vn.hyperparams.HyperParams(config, train_flag=train_flag)
        # init and run eng
        print('Creating network')
        self.model_name = hyper_params.model_name
        self.batch = hyper_params.batch
        self.use_cuda = hyper_params.cuda
        self.weights = hyper_params.weights
        self.conf_thresh = hyper_params.conf_thresh
        self.network_size = hyper_params.network_size
        self.labels = hyper_params.labels
        self.nworkers = hyper_params.nworkers
        self.pin_mem = hyper_params.pin_mem
        self.nms_thresh = hyper_params.nms_thresh
        self.results = hyper_params.results
        test_args = {'conf_thresh': self.conf_thresh, 'network_size': self.network_size, 'labels': self.labels}
        self.net = models.__dict__[self.model_name](hyper_params.classes, self.weights, train_flag=2, test_args=test_args)
        self.net.eval()
        #print('Net structure\n%s' % self.net)
        if self.use_cuda:
            self.net.cuda()
        self.loader = torch.utils.data.DataLoader(
            CustomDataset(hyper_params),
            batch_size=self.batch,
            shuffle=False,
            drop_last=False,
            num_workers=self.nworkers if self.use_cuda else 0,
            pin_memory=self.pin_mem if self.use_cuda else False,
            collate_fn=vn_data.list_collate,
        )
        print('Running network')
        self.fill_color = 127
        self.netw, self.neth = self.network_size
        #函数接口
        self.img = np.array(cv2.imread('./images/YDXJ0013_1399.jpg'))
        #self.img = self.img[:,:,::-1]


    def detTrepang(self):
        #要深度拷贝
        #im_show是BGR格式的
        im_show = copy.deepcopy(self.img)
        # 输入给算法模型的格式是RGB
        self.img = self.img[:, :, ::-1]
        self.Imgdata = to_tensor(self.tf_cv())
        self.im_data = torch.reshape(self.Imgdata,[1,3,self.netw,self.neth])
        anno, det = {}, {}
        if self.use_cuda:
            self.im_data = self.im_data.cuda()
        with torch.no_grad():
            self.output, _ = self.net(self.im_data, )
        key_val = len(anno)
        det.update({self.loader.dataset.keys[key_val + k]: v for k, v in enumerate(self.output)})
        reorg_dets = self.reorgDetection(det)
        image,numTre = self.genResults(reorg_dets,im_show)
        return image,numTre

    def tf_cv(self):
        """ Letterbox and image to fit in the network """
        im_h, im_w = self.img.shape[:2]
        self.orig_width = im_w
        self.orig_height = im_h
        if im_w == self.netw and im_h == self.neth:
            self.scale = None
            self.pad = None
            return self.img

        # Rescaling
        if im_w / self.netw >= im_h / self.neth:
            self.scale = self.netw / im_w
        else:
            self.scale = self.neth / im_h
        if self.scale != 1:
            self.img = cv2.resize(self.img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            im_h, im_w = self.img.shape[:2]

        if im_w == self.netw and im_h == self.neth:
            self.pad = None
            return self.img

        # Padding
        channels = self.img.shape[2] if len(self.img.shape) > 2 else 1
        pad_w = (self.netw - im_w) / 2
        pad_h = (self.neth - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        self.img = cv2.copyMakeBorder(self.img, self.pad[1], self.pad[3], self.pad[0], self.pad[2], cv2.BORDER_CONSTANT, value=(self.fill_color,)*channels)
        return self.img

    def reorgDetection(self,dets):  # , prefix):
        global reorg_dets
        reorg_dets = {}
        for k, v in dets.items():
            scale = min(float(self.netw) / self.orig_width, float(self.neth) / self.orig_height)
            new_width = self.orig_width * scale
            new_height = self.orig_height * scale
            pad_w = (self.netw - new_width) / 2.0
            pad_h = (self.neth - new_height) / 2.0

            for iv in v:
                xmin = iv.x_top_left
                ymin = iv.y_top_left
                xmax = xmin + iv.width
                ymax = ymin + iv.height
                conf = iv.confidence
                class_label = iv.class_label
                # print(xmin, ymin, xmax, ymax)

                xmin = max(0, float(xmin - pad_w) / scale)
                xmax = min(self.orig_width - 1, float(xmax - pad_w) / scale)
                ymin = max(0, float(ymin - pad_h) / scale)
                ymax = min(self.orig_height - 1, float(ymax - pad_h) / scale)

                reorg_dets.setdefault(class_label, {})
                reorg_dets[class_label].setdefault("frame", [])
                # line = '%s %f %f %f %f %f' % (name, conf, xmin, ymin, xmax, ymax)
                piece = (xmin, ymin, xmax, ymax, conf)
                reorg_dets[class_label]["frame"].append(piece)
        return reorg_dets

    def genResults(self, reorg_dets,im_show):
        results_folder = 'results'
        nms_thresh = 0.45
        thresh = 0.5
        global num_trepang
        num_trepang = 0
        for label, pieces in reorg_dets.items():
            for name in pieces.keys():
                pred = np.array(pieces[name], dtype=np.float32)
                keep = nms(pred,nms_thresh, force_cpu=True)
                # keep = soft_nms(pred, sigma=0.5, Nt=0.3, method=1)
                for ik in keep:
                    bbox = tuple(int(np.round(x)) for x in pred[ik][:4])
                    score = pred[ik][-1]
                    if score > thresh:
                        num_trepang = num_trepang +1
                        cv2.rectangle(im_show, bbox[0:2], bbox[2:4], (0, 0, 255), 2)
                        cv2.putText(im_show, '%s: %.3f' % ("trepang", score), (bbox[0], bbox[1] + 15),
                                            cv2.FONT_HERSHEY_PLAIN,
                                            1.0, (0, 255, 0), thickness=1)
        #result_path = os.path.join("./results/" + "det.jpg")
        #cv2.imwrite(result_path, im_show)
        return im_show,num_trepang

if __name__ == '__main__':
    det = VOCTest()
    det.detTrepang()
