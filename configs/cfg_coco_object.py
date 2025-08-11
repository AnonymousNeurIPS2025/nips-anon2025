_base_ = './base_config.py'

short_side = 480

# model settings
model = dict(
    name_path='./configs/cls_coco_object.txt',
    masks_path = None,
    alpha=0.5,  
    prob_thd=0.1,
)

# dataset settings
dataset_type = 'COCOObjectDataset'
data_root = './data/coco_object'

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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))