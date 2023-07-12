# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image
import os
import pyqrcode
import io
import torch

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules
import gradio as gr
from mmengine.logging import MMLogger


# test
class qrcode():

    def __init__(self) -> None:
        self.control = None
        self.generated_img = None


    def change(self, ratio=0.5):
        w = int(self.generated_img.shape[1] * ratio)
        c1 = self.control[:,:w,:]
        c2 = self.generated_img[:,w:,:]
        com_img = np.concatenate([c1, c2], axis=1)
        return com_img

    def composite(self, qrcode, generated_img, ratio=0.5):
        w = int(generated_img.shape[1] * ratio)
        c1 = qrcode[:,:w,:]
        c2 = generated_img[:,w:,:]
        com_img = np.concatenate([c1, c2], axis=1)
        return com_img
    
    def qrcode_generator(self,
                    text, 
                    prompt, 
                    negative_prompt,
                    num_inference_steps,
                    guidance_scale,
                    controlnet_conditioning_scale
                    ):
    
        qr= pyqrcode.create(text, error='H', version=6, mode='binary')
        qr.png('qrcode.png', scale=15)

        cfg = Config.fromfile('projects/QRcode_generator/controlnet-brightness.py')
        cfg.model.unet.from_pretrained = 'dreamlike-art/dreamlike-diffusion-1.0'
        cfg.model.vae.from_pretrained = 'dreamlike-art/dreamlike-diffusion-1.0'

        # controlnet config
        cfg.model.controlnet.attention_head_dim = 8
        cfg.model.controlnet.block_out_channels = [320,640,1280,1280]
        cfg.model.controlnet.conditioning_embedding_out_channels = [16,32,96,256]
        cfg.model.controlnet.controlnet_conditioning_channel_order = "rgb"
        cfg.model.controlnet.cross_attention_dim = 768
        cfg.model.controlnet.down_block_types = ["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"]
        cfg.model.controlnet.downsample_padding = 1
        cfg.model.controlnet.flip_sin_to_cos = True
        cfg.model.controlnet.freq_shift = 0
        cfg.model.controlnet.in_channels = 4
        cfg.model.controlnet.layers_per_block = 2
        cfg.model.controlnet.mid_block_scale_factor = 1
        cfg.model.controlnet.norm_eps = 1e-05
        cfg.model.controlnet.norm_num_groups = 32
        cfg.model.controlnet.only_cross_attention = False
        cfg.model.controlnet.resnet_time_scale_shift = "default"
        cfg.model.controlnet.sample_size = 32
        cfg.model.controlnet.upcast_attention = False
        cfg.model.controlnet.use_linear_projection = False
        cfg.model.controlnet.from_pretrained = 'ioclab/control_v1p_sd15_brightness'

        cfg.model.init_cfg['type'] = 'convert_from_unet'

        controlnet = MODELS.build(cfg.model)

        # call init_weights manually to convert weight
        controlnet.init_weights()

        prompt =  prompt
        negative_prompt = negative_prompt
        control_img = mmcv.imread('qrcode.png')
        control_img = cv2.resize(control_img, (256,256))
        control_img = control_img[:,:,0:1]
        control_img = np.concatenate([control_img]*3, axis=2)
        control = Image.fromarray(control_img)

        num_inference_steps = num_inference_steps
        guidance_scale = guidance_scale
        num_images_per_prompt = 1
        controlnet_conditioning_scale = controlnet_conditioning_scale
        height=256
        width=256
        
        output_dict = controlnet.infer(
                        prompt = prompt, 
                        control = control,
                        height = height,
                        width = width,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        num_inference_steps=num_inference_steps, 
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images_per_prompt,
                        negative_prompt=negative_prompt,
                        )

        generated_img  = output_dict['samples'][0]
        self.control = np.array(control)
        self.generated_img = np.array(generated_img)
        com_img = self.composite(self.control, self.generated_img)
    
        return com_img
    

if __name__ == '__main__':
    
    register_all_modules()
    generator = qrcode()

    with gr.Blocks() as demo:
        gr.Markdown('# MMagic QRcode generator Demo')
        gr.Markdown('This is a QRcode generator based on MMagic')
        text = gr.Textbox(label="QRcode text", placeholder='test')
        prompt= gr.Textbox(label="Prompt",placeholder='dreamlikeart, an zebra')
        negative_prompt= gr.Textbox(label="Negative Prompt",placeholder='ugly, bad quality')
        with gr.Row():
            num_inference_steps = gr.Slider(0, 10, value=5, step=1, label="Inference Steps")
            guidance_scale = gr.Slider(0, 10, value=7.5, step=0.5, label="Guidance Scale")
            controlnet_conditioning_scale = gr.Slider(0, 1, value=0.7, label="Controlnet Conditioning Scale")
        run_btn = gr.Button("Run")
        ratio = gr.Slider(0, 1, value=0.5, step=0.1)
        with gr.Row():
            o = gr.Image(shape=(256, 256))
        inputs = [text,prompt,negative_prompt,num_inference_steps,guidance_scale,controlnet_conditioning_scale]
        run_btn.click(fn= generator.qrcode_generator, inputs=inputs, outputs=o)
        ratio.change(fn = generator.change, inputs = ratio, outputs = o)

    demo.launch()
    

