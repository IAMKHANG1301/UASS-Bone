import sys
import torch
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, build_model
from detectron2.layers import ShapeSpec
from detectron2.config import get_cfg
from mask2former.config import add_maskformer2_config
from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0

@BACKBONE_REGISTRY.register()
class EfficientViTDetectron2Wrapper(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.bottom_up = efficientvit_backbone_b0(pretrained=True)
        self._out_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        self._out_feature_channels = {"stage0": 8, "stage1": 16, "stage2": 32, "stage3": 64, "stage4": 128}
        self._out_feature_strides = {"stage0": 2, "stage1": 4, "stage2": 8, "stage3": 16, "stage4": 32}
        
    def forward(self, x):
        outputs = self.bottom_up(x)
        ret = {}
        if isinstance(outputs, dict):
            for k in self._out_features:
                if k in outputs: ret[k] = outputs[k]
        elif isinstance(outputs, (list, tuple)):
            stage_names = ["stage0", "stage1", "stage2", "stage3", "stage4"]
            for i, feat in enumerate(outputs):
                if i < len(stage_names) and stage_names[i] in self._out_features:
                    ret[stage_names[i]] = feat
        return ret
        
    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }

def setup_config():
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
    cfg.MODEL.BACKBONE.NAME = "EfficientViTDetectron2Wrapper"
    
    cfg.DATASETS.TRAIN = ("fracatlas_train",)
    cfg.DATASETS.TEST = ("fracatlas_train",)
    
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1 
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["stage1", "stage2", "stage3", "stage4"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["stage2", "stage3", "stage4"]
    
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"
    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.0
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.PRE_NORM = False
    return cfg

def build_specialist_model(cfg):
    return build_model(cfg)