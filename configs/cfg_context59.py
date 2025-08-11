_base_ = './base_config.py'

short_side = 640

# model settings
model = dict(
    name_path='./configs/cls_context59.txt',
    masks_path = None,
    alpha=0.8,
)

# dataset settings
dataset_type = 'PascalContext59Dataset'
data_root = './data/VOCdevkit/VOC2010'

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=0.7, name='visualizer', dataset_name='pcontext',) # alpha=1.0 # https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py# update

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, short_side), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
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
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))