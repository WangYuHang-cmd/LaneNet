#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

# model = resnet50(pretrained=True)
# target_layers = [model.layer4[-1]]
# input_tensor = "/home/henry/Desktop/lanenet-lane-detection/data/training_data_example/image/LKA1.jpg"
# cam = GradCAM(model=model, target_layers=target_layers)
# targets = [ClassifierOutputTarget(281)]
# grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
# grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--with_lane_fit', type=args_str2bool, help='If need to do lane fit', default=True)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path, with_lane_fit=True):
    """

    :param image_path:
    :param weights_path:
    :param with_lane_fit:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    LOG.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    # binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
    binary_seg_ret, instance_seg_ret, cam = net.inference(input_tensor=input_tensor, target_class=1, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        loop_times = 500
        for i in range(loop_times):
            binary_seg_image, instance_seg_image, cam_image = sess.run(
                [binary_seg_ret, instance_seg_ret, cam],
                feed_dict={input_tensor: [image]}
            )
        t_cost = time.time() - t_start
        t_cost /= loop_times
        LOG.info('Single imgae  cost time: {:.5f}s'.format(t_cost))

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=with_lane_fit,
            data_source='tusimple'
        )
        mask_image = postprocess_result['mask_image']
        if with_lane_fit:
            lane_params = postprocess_result['fit_params']
            LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
            for i in range(len(lane_params)):
                LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))

        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        # plt.figure('cam_image')
        # cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_image[0]), cv2.COLORMAP_JET)
        # cam_heatmap = cv2.resize(cam_heatmap, (image_vis.shape[1], image_vis.shape[0]))
        # cam_overlaid = cv2.addWeighted(image_vis, 0.5, cam_heatmap, 0.5, 0)
        # plt.imshow(cam_overlaid[:, :, (2, 1, 0)])
        # plt.figure('cam_image')
        # # 添加阈值处理
        # threshold = 0.5
        # cam_image[cam_image < threshold] = 0
        
        # # 降低透明度和改变叠加方式
        # alpha = 0.3
        # cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_image[0]), cv2.COLORMAP_JET)
        # cam_heatmap = cv2.resize(cam_heatmap, (image_vis.shape[1], image_vis.shape[0]))
        # cam_overlaid = cv2.addWeighted(image_vis, 1-alpha, cam_heatmap, alpha, 0)

        # plt.imshow(cam_overlaid[:, :, (2, 1, 0)])
        # plt.axis('off')
        plt.figure('cam_image')
    
        # 更高的阈值过滤
        threshold = 0.7
        cam_image[cam_image < threshold] = 0
        
        # 使用更柔和的colormap和更低的透明度
        alpha = 0.2  # 降低透明度
        cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min())  # 归一化
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_image[0]), cv2.COLORMAP_HOT)  # 使用HOT colormap
        cam_heatmap = cv2.resize(cam_heatmap, (image_vis.shape[1], image_vis.shape[0]))
        
        # 更正维度不匹配问题
        cam_image_resized = cv2.resize(cam_image[0], (image_vis.shape[1], image_vis.shape[0]))
        mask = cam_image_resized > 0
        mask = np.stack([mask] * 3, axis=2)  # 扩展到3通道

        # 使用修正后的mask
        cam_overlaid = image_vis.copy()
        cam_overlaid[mask] = cv2.addWeighted(image_vis, 1-alpha, cam_heatmap, alpha, 0)[mask]
        
        plt.imshow(cam_overlaid[:, :, (2, 1, 0)])
        plt.axis('off')
        plt.show()

    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.image_path, args.weights_path, with_lane_fit=args.with_lane_fit)
