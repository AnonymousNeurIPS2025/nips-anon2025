import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.distributed as dist
import cv2
import numpy as np
from sklearn.cluster import SpectralClustering

import sys

sys.path.append("..")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from prompts.imagenet_template import openai_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS

@MODELS.register_module()
class DVGTSegmentation(BaseSegmentor):
    def __init__(self, 
                 clip_type='openai',
                 model_type='convnext_l',
                 name_path=None, 
                 device=torch.device('cuda'), 
                 prob_thd=0.0, 
                 logit_scale=40, 
                 slide_stride=0, 
                 slide_crop=0,
                 clip_resolution=(640, 640),
                 area_thd=None,
                 masks_path=None,
                 sam_checkpoint_path="./model/sam_vit_h_4b8939.pth",
                 max_n_cluster=32,
                 simthresh=0.1,
                 alpha=0.8, 
                 dino_model='dino_vitb8',
                 ):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)
        
        self.masks_path = masks_path

        self.tokenizer = None
        self.clip_type = clip_type
        self.clip_resolution = clip_resolution
        
        if self.clip_type == 'openai':
            import open_clip
            from open_clip.transformer import text_global_pool
            
            if "convnext" in model_type:
                if "b" in model_type or "base" in model_type:
                    name, pretrain = ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k')
                elif "l" in model_type or "large" in model_type:
                    name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
                else:
                    assert False, f"clip_pretrained must contain 'base'/'b' or 'large'/'l', but got {model_type}"

                clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrain, device=device)
                self.tokenizer = open_clip.get_tokenizer(name)
                
                def custom_image(self, image, dense=False, normalize=False):
                    if dense:
                        features = self.visual.trunk.forward_features(image)
                        features = self.visual.trunk.head.norm(features)
                        features = self.visual.head(features.permute(0, 2, 3, 1))
                    else:
                        features = self.visual(image)
                    return F.normalize(features, dim=-1) if normalize else features
                
                funcType = type(clip_model.encode_image)
                clip_model.encode_image = funcType(custom_image, clip_model)
                
                def custom_text(self, text, normalize: bool = False, prompt = None):
                    cast_dtype = self.transformer.get_cast_dtype()

                    if prompt is not None:
                        x = prompt
                    else:
                        x = self.token_embedding(text).to(cast_dtype) 

                    x = x + self.positional_embedding.to(cast_dtype)
                    x = self.transformer(x, attn_mask=self.attn_mask)
                    x = self.ln_final(x)  
                    x, _ = text_global_pool(x, text, self.text_pool_type)
                    if self.text_projection is not None:
                        if isinstance(self.text_projection, nn.Linear):
                            x = self.text_projection(x)
                        else:
                            x = x @ self.text_projection

                    return F.normalize(x, dim=-1) if normalize else x
                
                funcType = type(clip_model.encode_text)
                clip_model.encode_text = funcType(custom_text, clip_model)
            else:
                from third_party import clip
                clip_model, clip_preprocess = clip.load(model_type, device=device, jit=False)
                self.tokenizer = clip.tokenize
        else:
            raise NotImplementedError

        self.clip_model = clip_model.float()
        # self.clip_preprocess = clip_preprocess
                
        for p in self.clip_model.parameters():
            p.requires_grad = False

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device) 
                feature = self.clip_model.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach() 

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        
        self.area_thd = area_thd
        self.device = device
        
        # SAM and clustering parameters
        self.sam_checkpoint_path = sam_checkpoint_path
        self.max_n_cluster = max_n_cluster
        self.simthresh = simthresh
        self.alpha = alpha
        self.dino_model = dino_model
        
        # Initialize SAM model (lazy loading)
        self.sam = None
        self.generator = None
        
        # Initialize DINO model (lazy loading)
        self.dino = None
        
        # Pixel normalization constants
        self.pixel_mean = torch.Tensor((123.675, 116.280, 103.530)).view(-1, 1, 1)  # RGB
        self.pixel_std = torch.Tensor((58.395, 57.120, 57.375)).view(-1, 1, 1)
        self.clip_pixel_mean = torch.Tensor((122.7709383, 116.7460125, 104.09373615)).view(-1, 1, 1)
        self.clip_pixel_std = torch.Tensor((68.5005327, 66.6321579, 70.3231630)).view(-1, 1, 1)
        
    def load_sam_model(self):
        """Load SAM model if not already loaded"""
        if self.sam is None:
            self.sam = sam_model_registry["default"](checkpoint=self.sam_checkpoint_path)
            self.sam.to(device=self.device)
            self.generator = SamAutomaticMaskGenerator(self.sam, stability_score_thresh=0.9)
        return self.generator
    
    def load_dino_model(self):
        """Load DINO model if not already loaded"""
        if self.dino is None:
            self.dino = torch.hub.load('facebookresearch/dino:main', self.dino_model)
            for p in self.dino.parameters():
                p.requires_grad = False
            self.dino.eval().to(self.device)
        return self.dino
    
    def resize_to_shorter_side(self, image, target_size=480):
        """Resize image keeping the aspect ratio by setting shorter side to target_size"""
        h, w = image.shape[:2]
        if h < w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))
        return cv2.resize(image, (new_w, new_h))
    
    def generate_sc_masks(self, image):
        """Generate spectral clustering masks using SAM and DINO features"""
        try:
            # Load models
            generator = self.load_sam_model()
            dino = self.load_dino_model()
            
            # Convert tensor to numpy if needed
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:  # batch dimension
                    image = image.squeeze(0)
                if image.dim() == 3 and image.shape[0] == 3:  # CHW format
                    image = image.permute(1, 2, 0)
                image = image.cpu().numpy()
                # Denormalize if needed - convert from [-1,1] or [0,1] to [0,255]
                if image.min() < 0:  # Assume normalized to [-1,1] or similar
                    image = (image + 1) * 127.5
                elif image.max() <= 1.0:  # Assume normalized to [0,1]
                    image = image * 255
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Ensure RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already RGB, good
                pass
            else:
                raise ValueError(f"Expected RGB image, got shape: {image.shape}")
            
            # Resize image
            resized_image = self.resize_to_shorter_side(image, target_size=480)
            
            # Generate SAM masks
            masks = generator.generate(resized_image)
            segs = [mask['segmentation'] for mask in masks]
            
            if len(segs) < self.max_n_cluster:
                # If not enough masks, just return sorted masks
                sc_masks = sorted(segs, key=lambda m: np.sum(m), reverse=True)
                return torch.stack([torch.tensor(s) for s in sc_masks], dim=0).to(self.device)
            
            # Prepare image tensor for feature extraction
            image_tensor = torch.Tensor(resized_image).permute(2, 0, 1)
            norm_image = (image_tensor - self.pixel_mean.cpu()) / self.pixel_std.cpu()
            resized_norm_image = F.interpolate(norm_image[None], size=(640, 640), mode='bilinear', align_corners=False).to(self.device)
            
            # Prepare masks
            mask_resolution = (256, 256)
            segs_tensor = torch.from_numpy(np.stack(segs)).float()
            masks_tensor = F.interpolate(segs_tensor[None], size=mask_resolution, mode='nearest').to(self.device).squeeze(0)
            
            # CLIP feature extraction
            clip_norm_image = (image_tensor - self.clip_pixel_mean.cpu()) / self.clip_pixel_std.cpu()
            resized_clip_norm_image = F.interpolate(clip_norm_image[None], size=(640, 640), mode='bilinear', align_corners=False).to(self.device)
            clip_features = self.clip_model.encode_image(resized_clip_norm_image, dense=True)
            clip_features /= clip_features.norm(dim=-1, keepdim=True)
            clip_features = clip_features.permute(0, 3, 1, 2)
            clip_features = F.interpolate(clip_features, size=mask_resolution, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
            clip_features = clip_features.detach()
            
            # Mask pooling and logits
            denorm = masks_tensor.sum(dim=[1, 2], keepdim=True) + 1e-8
            mask_features = torch.einsum("hwc, nhw -> nc", clip_features, masks_tensor / denorm)
            masks_logit = torch.matmul(mask_features, self.query_features.T)
            
            # Apply similarity threshold
            mask = masks_logit < self.simthresh
            masks_logit = masks_logit.masked_fill(mask, 1e-5)
            norm = masks_logit.norm(dim=-1, keepdim=True).clamp(min=1e-5)
            masks_logit_norm = masks_logit / norm
            
            clip_sim = torch.matmul(masks_logit_norm, masks_logit_norm.T)
            inter_adj = torch.clamp(clip_sim, min=1e-5, max=1.0).cpu().numpy()
            
            # DINO feature extraction
            with torch.no_grad():
                dino_feature = dino.get_intermediate_layers(resized_norm_image)[0] 
                patch_size = dino.patch_embed.patch_size
                bs, h, w = resized_norm_image.shape[0], resized_norm_image.shape[-2] // patch_size, resized_norm_image.shape[-2] // patch_size
                feats = dino_feature[:, 1:, :].reshape(bs, h, w, -1).permute(0, 3, 1, 2)
            
            dino_features = F.interpolate(feats, size=mask_resolution, mode='bilinear', align_corners=True).permute(0, 2, 3, 1).squeeze(0)
            denorm = masks_tensor.sum(dim=[1, 2], keepdim=True) + 1e-8
            dino_mask_features = torch.einsum("hwc, nhw -> nc", dino_features, masks_tensor / denorm)
            dino_mask_features = F.normalize(dino_mask_features, dim=-1)
            dino_feature_sim = torch.matmul(dino_mask_features, dino_mask_features.T)
            dino_feature_sim = torch.clamp(dino_feature_sim, min=1e-5, max=1.0)
            
            intra_adj = dino_feature_sim.cpu().numpy()
            
            # Combine adjacency matrices
            adj_mat = self.alpha * intra_adj + (1 - self.alpha) * inter_adj
            
            # Spectral clustering
            max_k = self.max_n_cluster if len(segs) > self.max_n_cluster else len(segs)
            k = min(max_k, len(segs))
            
            sc = SpectralClustering(k, affinity='precomputed', n_init=100)
            sc.fit(adj_mat)
            labels = sc.labels_
            idx2label = {i: l for i, l in enumerate(labels)}
            
            # Create clustered masks
            sc_masks = [None for _ in range(k)]
            for i, seg in enumerate(segs):
                label = idx2label[i]
                if sc_masks[label] is None:
                    sc_masks[label] = seg
                else:
                    sc_masks[label] = np.maximum(sc_masks[label], seg)
            
            sc_masks = [mask for mask in sc_masks if mask is not None]
            # sc_masks = sorted(sc_masks, key=lambda m: np.sum(m), reverse=True)
            
            return torch.stack([torch.tensor(s) for s in sc_masks], dim=0).to(self.device)
        
        except Exception as e:
            print(f"Error in generate_sc_masks: {e}")
            # Return a simple fallback mask (whole image)
            if isinstance(image, torch.Tensor):
                h, w = image.shape[-2:]
            else:
                h, w = image.shape[:2]
            fallback_mask = torch.ones(1, h, w, dtype=torch.bool).to(self.device)
            return fallback_mask
        
    @torch.no_grad()
    def forward_feature(self, img, logit_size=None, obj_masks=None):
        if type(img) == list:
            img = img[0]
        
        clip_img = F.interpolate(img, size=self.clip_resolution, mode='bilinear', align_corners=True)
        # clip_img = img
        
        if self.clip_type == 'openai':
            image_features = self.clip_model.encode_image(clip_img, dense=True) # b h w c
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.permute(0, 3, 1, 2) # 
        elif self.clip_type == 'sclip':
            image_features = self.clip_model.encode_image(clip_img, return_all=True, csa=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features[:, 1:]
            patch_size = self.clip_model.visual.patch_size
            b, hw, c = image_features.shape
            h, w = clip_img.shape[-2] // patch_size, clip_img.shape[-1] // patch_size
            image_features = image_features.permute(0, 2, 1).reshape(b, c, h, w) # 
        
        # image_features = image_features.permute(0, 3, 1, 2) # b c h w
        image_features = image_features.detach()
        
        if logit_size == None:
            image_features = nn.functional.interpolate(image_features, size=img.shape[-2:], mode='bilinear', align_corners=False)
        else:
            image_features = nn.functional.interpolate(image_features, size=logit_size, mode='bilinear', align_corners=False)

        b, _, h, w = image_features.shape
        # logits = torch.einsum('b c h w, n c -> b h w n', image_features, self.query_features) # logit p(b h w n)
        logits = torch.zeros(b, h, w, self.query_features.shape[0]).to(image_features.device).type(image_features.dtype) # 

        pred_fg = torch.zeros(logits.shape[1:3]).type(torch.bool).to(logits.device)
        for i in range(obj_masks.shape[0]):
            if obj_masks[i].sum() == 0: continue
            mask_feat = torch.sum(image_features * obj_masks[i], dim=[-1, -2]) / obj_masks[i].sum()
            assert image_features.shape[0] == 1, 'use repeat instead of unsqueeze(0)'
            logits[obj_masks[i].unsqueeze(0)] = mask_feat @ self.query_features.T
            pred_fg[obj_masks[i]] = True

        return logits.permute(0, 3, 1, 2)
    
    def forward_slide(self, img, img_metas, stride=112, crop_size=224, obj_masks=None): # todos. modify the forward_slide function
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                
                if obj_masks.shape[0]:
                    obj_masks = nn.functional.interpolate(obj_masks.unsqueeze(0).type(torch.float32), size=(h_img, w_img), mode='nearest').squeeze(0).type(torch.bool)  # avoid mask shirnkage
                    crop_obj_masks = obj_masks[:, y1:y2, x1:x2]
                else:
                    crop_obj_masks = torch.zeros(0, *crop_img.shape[2:]).type(torch.bool)

                if crop_img.shape[-2:] != crop_obj_masks.shape[-2:]: # debug
                    print(crop_img.shape, crop_obj_masks.shape); import pdb ; pdb.set_trace()
                    h, w = crop_img.shape[-2:]
                    m, ch, cw = crop_obj_masks.shape
                    _crop_obj_masks = torch.zeros(m, h, w).type(crop_obj_masks.dtype).to(crop_obj_masks.device)
                    _crop_obj_masks[:, :ch, :cw] = crop_obj_masks
                    crop_obj_masks = _crop_obj_masks

                crop_seg_logit = self.forward_feature(crop_img, obj_masks=crop_obj_masks).detach()
                
                torch.cuda.empty_cache()
                
                preds += nn.functional.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')
        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        
        if self.masks_path is None:
            input_image = inputs[0]  # Assume batch size 1
            sc_masks_tensor = self.generate_sc_masks(input_image)
            
            # Sort by area (ascending)
            areas = sc_masks_tensor.sum(dim=[1, 2])
            sorted_indices = torch.argsort(areas)
            sc_masks = sc_masks_tensor[sorted_indices]
            
            # Resize masks to original image shape
            sc_masks = F.interpolate(sc_masks.unsqueeze(1).float(), size=batch_img_metas[0]['ori_shape'], mode='nearest').squeeze(1).bool()
        else:
            fname = batch_img_metas[0]['img_path'].split('/')[-1].split('.')[0]
            import os
            sc_masks = torch.load(os.path.join(self.masks_path, f'{fname}.pth'))
            
            areas = sc_masks.sum(dim=[1, 2])
            sorted_indices = torch.argsort(areas) 
            sc_masks = sc_masks[sorted_indices]
            sc_masks = F.interpolate(sc_masks.unsqueeze(1).float(), size=batch_img_metas[0]['ori_shape'], mode='nearest').squeeze(1).bool()

        # if self.slide_crop > 0:
        #     seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop, sc_masks)
        # else:
        #     seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'], sc_masks)
        seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'], sc_masks)

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                
            if self.area_thd is not None:
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(seg_logits.dtype)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)  # prone background
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype)
                seg_logits[1:] *= area_pred.transpose(0, -1)

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0 # debugging 할때 이걸 1로 둬보자.

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples
    

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices