#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午8:50
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet.py
# @IDE: PyCharm
"""
Implement LaneNet Model
"""
import tensorflow as tf

from lanenet_model import lanenet_back_end
from lanenet_model import lanenet_front_end
from semantic_segmentation_zoo import cnn_basenet


class LaneNet(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, cfg):
        """

        """
        super(LaneNet, self).__init__()
        self._cfg = cfg
        self._net_flag = self._cfg.MODEL.FRONT_END

        self._frontend = lanenet_front_end.LaneNetFrondEnd(
            phase=phase, net_flag=self._net_flag, cfg=self._cfg
        )
        self._backend = lanenet_back_end.LaneNetBackEnd(
            phase=phase, cfg=self._cfg
        )

    def inference(self, input_tensor, target_class, name, reuse=False):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # 提取特征
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=reuse
            )

            # 获取最后的卷积层和分割预测
            final_conv = extract_feats_result['final_conv']['data']
            binary_seg_logits = extract_feats_result['binary_segment_logits']['data']

            # 计算Grad-CAM
            target_scores = binary_seg_logits[..., target_class]
            gradients = tf.gradients(target_scores, final_conv)[0]
            weights = tf.reduce_mean(gradients, axis=(1, 2))
            cam = tf.einsum('bhwc,bc->bhw', final_conv, weights)
            cam = tf.nn.relu(cam)
            cam = tf.image.resize_bilinear(
                tf.expand_dims(cam, -1),
                input_tensor.shape[1:3]
            )

            # 正常推理
            binary_seg_prediction, instance_seg_prediction = self._backend.inference(
                binary_seg_logits=binary_seg_logits,
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                name='{:s}_backend'.format(self._net_flag),
                reuse=reuse
            )

            return binary_seg_prediction, instance_seg_prediction, cam

    # def inference(self, input_tensor, name, reuse=False):
    #     """

    #     :param input_tensor:
    #     :param name:
    #     :param reuse
    #     :return:
    #     """
    #     with tf.variable_scope(name_or_scope=name, reuse=reuse):
    #         # first extract image features
    #         extract_feats_result = self._frontend.build_model(
    #             input_tensor=input_tensor,
    #             name='{:s}_frontend'.format(self._net_flag),
    #             reuse=reuse
    #         )

    #         # second apply backend process
    #         binary_seg_prediction, instance_seg_prediction = self._backend.inference(
    #             binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
    #             instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
    #             name='{:s}_backend'.format(self._net_flag),
    #             reuse=reuse
    #         )

    #     return binary_seg_prediction, instance_seg_prediction

    def compute_loss(self, input_tensor, binary_label, instance_label, name, reuse=False):
        """
        calculate lanenet loss for training
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=reuse
            )

            # second apply backend process
            calculated_losses = self._backend.compute_loss(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                binary_label=binary_label,
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                instance_label=instance_label,
                name='{:s}_backend'.format(self._net_flag),
                reuse=reuse
            )

        return calculated_losses


    def compute_gradcam(self, input_tensor, target_class, target_layer='final_conv', name='lanenet', reuse=False):
        """
        Compute Grad-CAM for LaneNet
        Args:
            input_tensor: Input image tensor, shape [batch_size, height, width, channels]
            target_class: Target class index (0: background, 1: lane)
            target_layer: Name of target conv layer for Grad-CAM
            name: Model name scope
            reuse: Whether to reuse variables
        Returns:
            cam: Class activation map
            logits: Model prediction logits
        """
        with tf.variable_scope(name, reuse=reuse):
            # Get model features and predictions
            feats = self._frontend.build_model(
                input_tensor=input_tensor,  
                name=f'{self._net_flag}_frontend',
                reuse=reuse
            )
            
            # Get target conv layer and prediction logits
            conv_output = feats[target_layer]['data']  
            pred_logits = feats['binary_segment_logits']['data']
            
            # Get target class score (using max to handle batch dimension)
            target_score = tf.reduce_max(pred_logits[..., target_class])
            # target_score = tf.reduce_mean(pred_logits[..., target_class])
            # target_score = pred_logits[..., target_class]
            print(target_score)
            
            # Compute gradients of score w.r.t. conv output
            # gradients = tf.gradients(target_score, conv_output)[0]
            gradients = tf.gradients(target_score, conv_output)[0]
            
            # Global average pooling on gradients
            weights = tf.reduce_mean(gradients, axis=(1, 2)) 
            
            # Compute weighted sum of conv outputs
            cam = tf.einsum('bhwc,bc->bhw', conv_output, weights)
            
            # Apply ReLU 
            cam = tf.nn.relu(cam)
            
            # Resize CAM to input size
            cam = tf.image.resize_bilinear(
                tf.expand_dims(cam, -1),
                input_tensor.shape[1:3]
            )
            
            return cam, pred_logits