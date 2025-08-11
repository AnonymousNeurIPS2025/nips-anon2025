_base_ = './base_config.py'

short_side = 480 # 336 448

# model settings
model = dict(
    name_path='./configs/cls_voc20.txt',
    masks_path = None,
    alpha=0.8,
    clip_resolution=(480, 480),
)

# dataset settings
dataset_type = 'PascalVOC20Dataset'
data_root = './data/VOCdevkit/VOC2012' #

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=0.7, name='visualizer', dataset_name='pascal_voc',) # alpha=1.0

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, short_side), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))