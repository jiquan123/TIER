import torch
import timm
from timm.models.vision_transformer import Block

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

class ViTExtractor(torch.nn.Module):
    def __init__(self):
        super(ViTExtractor, self).__init__()
        #self.vit = timm.create_model('vit_large_patch16_224', pretrained=True, pretrained_cfg_overlay=dict(file='/root/autodl-tmp/yjq/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'))
        self.vit = timm.create_model('vit_large_patch16_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

    def extract_feature(self, save_output):
        x = save_output.outputs[-1][:, 0]
        return x

    def forward(self, x):
        _x = self.vit(x)
        features = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()
        return features

class SwinExtractor(torch.nn.Module):
    def __init__(self):
        super(SwinExtractor, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.model.forward_features(x)
        features = self.pool(features.permute(0, 3, 1, 2)).squeeze(2).squeeze(2)
        return features
    

def MAE():
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True)

    # 移除分类头部（将其置换为 Identity 以返回 Encoder 的输出）
    model.head = torch.nn.Identity()

    return model

