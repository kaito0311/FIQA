
import cv2 
import onnx
import torch 
import numpy as np 
import onnxruntime
from typing import Any


class ONNX_IMINT:
    def __init__(self, path="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/stacking_avg_r160+ada-unnorm-stacking-ada-1.6.onnx") -> None:
        self.session = onnxruntime.InferenceSession(path, providers = ['CUDAExecutionProvider'])

        self.input_session = self.session.get_inputs() 
        self.input_name = self.input_session[0].name 

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        IN_IMAGE_H = self.session.get_inputs()[0].shape[2]
        IN_IMAGE_W = self.session.get_inputs()[0].shape[3]
        if isinstance(x, torch.Tensor):
            x_numpy = x.clone().detach().cpu().numpy() 
        assert IN_IMAGE_H == x_numpy.shape[2] 

        outputs = self.session.run(None, {self.input_name: x_numpy})
        return outputs[0]


