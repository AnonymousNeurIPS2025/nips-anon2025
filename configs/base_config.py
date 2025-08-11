# base configurations
model = dict(
    type='DVGTSegmentation',
    clip_type='openai',
    model_type = 'convnext-l', # convnext-l ViT-B/16 ViT-L/14 ViT-L/14@336px convnext-b 
    slide_crop=0, # 480 # if you want to use sliding window, set the slide_stride and slide_crop to 480 and 160 respectively
    slide_stride=0, # 160
    clip_resolution=(640, 640),
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=0.7, name='visualizer') # alpha=1.0

# https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317 참고하여 visualizer에 dataset_name 설정

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1)) # interval=5로 설정되어 있었음