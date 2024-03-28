
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
class ModelIntermFeatPatcher:

    def patch(self, model, no_head=False, profiling=0):
        model.profiling = profiling
        model.forward_features = types.MethodType(self.__class__.forward_features, model)
        model.forward_head = types.MethodType(self.__class__.forward_head, model)
        if profiling > 0:
            model.forward = types.MethodType(self.__class__.forward_features_profiling, model)
        if no_head:
            model.forward = model.forward_features
            
            return model
        return model


class ViTPatcher(ModelIntermFeatPatcher):

    def forward_features(self, x):
        interm_feat = []
        interm_feat.append(x)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
            interm_feat.append(x)
        x = self.norm(x)
        return x, interm_feat

    def forward_head(self, x, pre_logits: bool = False):
        if isinstance(x, tuple):
            x = x[0]
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        if isinstance(x, tuple):
            return x if pre_logits else self.head(x), x[1]
        return x if pre_logits else self.head(x)


class DaViTPatcher(ModelIntermFeatPatcher):

    def forward_features(self, x):
        interm_feat = []
        interm_feat.append(x)
        x = self.stem(x)
        for stage in self.stages:
            x = stage.downsample(x)
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
        x = self.norm_pre(x)
        return x, interm_feat

    def forward_features_profiling(self, x):
        interm_feat = []
        interm_feat.append(x)
        if len(interm_feat) >= self.profiling:
            return x, interm_feat
        x = self.stem(x)
        for stage in self.stages:
            x = stage.downsample(x)
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
                if len(interm_feat) >= self.profiling:
                    return x, interm_feat
        x = self.norm_pre(x)
        return x, interm_feat

    def forward_head(self, x, pre_logits: bool = False):
        x_ = x
        if isinstance(x, tuple):
            x = x[0]
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        if isinstance(x_, tuple):
            return x if pre_logits else self.head.fc(x), x_[1]
        return x if pre_logits else self.head.fc(x)


class SwinTPatcher(ModelIntermFeatPatcher):

    def forward_head(self, x, pre_logits: bool = False):
        x_ = x
        if isinstance(x, tuple):
            x = x[0]
        return self.head(x, pre_logits=True) if pre_logits else self.head(x), x_[1]

    def forward_features(self, x):
        interm_feat = []
        interm_feat.append(x)
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer.downsample(x)
            for blk in layer.blocks:
                x = blk(x)
                interm_feat.append(x.permute([0, 3, 1, 2]))
        x = self.norm(x)
        return x, interm_feat

    def forward_features_profiling(self, x):
        interm_feat = []
        interm_feat.append(x)
        if len(interm_feat) >= self.profiling:
            return x, interm_feat
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer.downsample(x)
            for blk in layer.blocks:
                x = blk(x)
                interm_feat.append(x.permute([0, 3, 1, 2]))
                if len(interm_feat) >= self.profiling:
                    return x, interm_feat
        x = self.norm(x)
        return x, interm_feat


class MaxVitPatcher(ModelIntermFeatPatcher):

    def forward_head(self, x, pre_logits: bool = False):
        x_ = x
        if isinstance(x, tuple):
            x = x[0]
        return self.head(x, pre_logits=pre_logits), x_[1]

    def forward_features(self, x):
        interm_feat = []
        interm_feat.append(x)
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
        x = self.norm(x)
        return x, interm_feat

    def forward_features_profiling(self, x):
        interm_feat = []
        interm_feat.append(x)
        if len(interm_feat) >= self.profiling:
            return x, interm_feat
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
                if len(interm_feat) >= self.profiling:
                    return x, interm_feat
        x = self.norm(x)
        return x, interm_feat


class MobileNetv3Patcher(ModelIntermFeatPatcher):

    def forward_features_profiling(self, x):
        interm_feat = []
        interm_feat.append(x)
        if len(interm_feat) >= self.profiling:
            return x, interm_feat

        x = self.conv_stem(x)
        x = self.bn1(x)
        for stage in self.blocks:
            for blk in stage:
                x = blk(x)
                interm_feat.append(x)
                if len(interm_feat) >= self.profiling:
                    return x, interm_feat
        x = self.norm_pre(x)
        return x, interm_feat

    def forward_features(self, x):
        interm_feat = []
        interm_feat.append(x)
        x = self.conv_stem(x)
        x = self.bn1(x)
        for stage in self.blocks:
            for blk in stage:
                x = blk(x)
                interm_feat.append(x)
        return x, interm_feat

    def forward_head(self, x, pre_logits: bool = False):
        x_ = x
        if isinstance(x, tuple):
            x = x[0]
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if pre_logits:
            return x, x_[1]
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x), x_[1]


class EfficientNetPatcher(MobileNetv3Patcher):

    def forward_head(self, x, pre_logits: bool = False):
        x_ = x
        if isinstance(x, tuple):
            x = x[0]
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.global_pool(x)
        if pre_logits:
            return x, x_[1]
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x), x_[1]


class ConvNextPatcher(ModelIntermFeatPatcher):

    def forward_features_profiling(self, x):
        interm_feat = []
        interm_feat.append(x)
        if len(interm_feat) >= self.profiling:
            return x, interm_feat

        x = self.stem(x)
        for stage in self.stages:
            x = stage.downsample(x)
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
                if len(interm_feat) >= self.profiling:
                    return x, interm_feat
        x = self.norm_pre(x)
        return x, interm_feat

    def forward_features(self, x):
        interm_feat = []
        interm_feat.append(x)

        x = self.stem(x)
        for stage in self.stages:
            x = stage.downsample(x)
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
        x = self.norm_pre(x)
        return x, interm_feat

    def forward_head(self, x, pre_logits: bool = False):
        x_ = x
        if isinstance(x, tuple):
            x = x[0]
        return  self.head(x, pre_logits=True) if pre_logits else self.head(x), x_[1]

class ResNetPatcher(ModelIntermFeatPatcher):

    def forward_features_profiling(self, x):
        interm_feat = []
        interm_feat.append(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        if len(interm_feat) >= self.profiling:
            return None, interm_feat
        for block in self.layer1:
            x = block(x)
            interm_feat.append(x)
            if len(interm_feat) >= self.profiling:
                return None, interm_feat
        for block in self.layer2:
            x = block(x)
            interm_feat.append(x)
            if len(interm_feat) >= self.profiling:
                return None, interm_feat
        for block in self.layer3:
            x = block(x)
            interm_feat.append(x)
            if len(interm_feat) >= self.profiling:
                return None, interm_feat
        for block in self.layer4:
            x = block(x)
            interm_feat.append(x)
            if len(interm_feat) >= self.profiling:
                return None, interm_feat

        return x, interm_feat

    def forward_features(self, x):
        interm_feat = []
        interm_feat.append(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        for block in self.layer1:
            x = block(x)
            interm_feat.append(x)
        for block in self.layer2:
            x = block(x)
            interm_feat.append(x)
        for block in self.layer3:
            x = block(x)
            interm_feat.append(x)
        for block in self.layer4:
            x = block(x)
            interm_feat.append(x)

        return x, interm_feat


    def forward_head(self, x, pre_logits: bool = False):
        x_ = x
        if isinstance(x, tuple):
            x = x[0]
        x = self.global_pool(x)
        return x if pre_logits else self.fc(x), x_[1]

class EfficientViTPatcher(ModelIntermFeatPatcher):
    # efficientvit_b1.r224_in1k
    # efficientvit_b2.r224_in1k
    # efficientvit_b3.r288_in1k
    def forward_features_profiling(self, x):
        interm_feat = []
        interm_feat.append(x)
        if len(interm_feat) >= self.profiling:
            return x, interm_feat

        if hasattr(self, 'patch_embed'):
            x = self.patch_embed(x)
        else:
            x = self.stem(x)
        for stage in self.stages:
            if hasattr(stage, 'downsample'):
                x = stage.downsample(x)
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
                if len(interm_feat) >= self.profiling:
                    return x, interm_feat
        return x, interm_feat

    def forward_features(self, x):
        interm_feat = []
        interm_feat.append(x)
        if hasattr(self, 'patch_embed'):
            x = self.patch_embed(x)
        else:
            x = self.stem(x)
        for stage in self.stages:
            if hasattr(stage, 'downsample'):
                x = stage.downsample(x)
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
        return x, interm_feat

    def forward_head(self, x, pre_logits: bool = False):
        x_ = x
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(self.global_pool, torch.nn.Module):
            x = self.global_pool(x)
        return self.head(x), x_[1]


class EfficientFormerv2Patcher(ModelIntermFeatPatcher):
    # efficientvit_b1.r224_in1k
    # efficientvit_b2.r224_in1k
    # efficientvit_b3.r288_in1k
    def forward_features_profiling(self, x):
        interm_feat = []
        interm_feat.append(x)
        if len(interm_feat) >= self.profiling:
            return x, interm_feat
        x = self.stem(x)
        for stage in self.stages:
            x = stage.downsample(x)
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
                if len(interm_feat) >= self.profiling:
                    return x, interm_feat
        x = self.norm(x)
        return x, interm_feat

    def forward_features(self, x):
        interm_feat = []
        interm_feat.append(x)
        x = self.stem(x)
        for stage in self.stages:
            x = stage.downsample(x)
            for blk in stage.blocks:
                x = blk(x)
                interm_feat.append(x)
        x = self.norm(x)
        return x, interm_feat

    def forward_head(self, x, pre_logits: bool = False):
        x_ = x
        if isinstance(x, tuple):
            x = x[0]

        if self.global_pool == 'avg':
            x = x.mean(dim=(2, 3))
        x = self.head_drop(x)
        if pre_logits:
            return x, x_[1]
        x, x_dist = self.head(x), self.head_dist(x)
        return (x + x_dist) / 2, x_[1]
