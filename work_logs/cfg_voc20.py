data_root = './data/VOCdevkit/VOC2012'
dataset_type = 'PascalVOC20Dataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=10, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(interval=1, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    alpha=0.8,
    clip_resolution=(
        480,
        480,
    ),
    clip_type='openai',
    masks_path=None,
    model_type='convnext-l',
    name_path='./configs/cls_voc20.txt',
    slide_crop=0,
    slide_stride=0,
    type='DVGTSegmentation')
resume = False
short_side = 480
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root='./data/VOCdevkit/VOC2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                480,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalVOC20Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        480,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    alpha=0.7,
    dataset_name='pascal_voc',
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_logs/'
