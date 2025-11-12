_base_ = 'E:\PythonProject\mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_ciou_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
        )
    ),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
)

data_root = r'E:\PythonProject\mmdetection\data\VisDrone/'
metainfo = {
    'classes': ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'),
    'palette': [
        (220, 20, 60),    # 行人 - 红色
        (255, 0, 0),      # 人群 - 纯红色
        (0, 255, 0),      # 自行车 - 绿色
        (0, 0, 255),      # 汽车 - 蓝色
        (255, 165, 0),    # 货车 - 橙色
        (128, 0, 128),    # 卡车 - 紫色
        (255, 192, 203),  # 三轮车 - 粉色
        (255, 215, 0),    # 带篷三轮车 - 金色
        (0, 255, 255),    # 公交车 - 青色
        (255, 0, 255)     # 摩托车 - 洋红色
    ]
}

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/annotation_train.json',
        data_prefix=dict(img='train/images/'),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/annotation_val.json',
        data_prefix=dict(img='val/images/'),
    )
)
test_dataloader = val_dataloader


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/annotation_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,
    val_interval=1)  # 验证间隔。每个 epoch 验证一次
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type="AmpOptimWrapper",   # 'OptimWrapper',  'AmpOptimWrapper'
    optimizer=dict(
        type='SGD',
        lr=0.0025,
        momentum=0.9,
        weight_decay=0.0001
    ),
    clip_grad=None,
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=500),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,         # 每个epoch都检查一次是否最佳
        save_best='auto',   # 核心参数：自动选择最佳模型
        rule='greater',     # 指标越大越好
        max_keep_ckpts=2    # 只保留最近2个最佳模型，避免磁盘爆炸
    ),
    visualization=dict(type='DetVisualizationHook')
)

log_level = 'INFO'  # 日志等级
load_from = None
resume = False  # 是否从 `load_from` 中定义的检查点恢复。 如果 `load_from` 为 None，它将恢复 `work_dir` 中的最新检查点。

# python tools/train.py E:\PythonProject\mmdetection\configs\faster_rcnn\faster-rcnn_r50_fpn_ciou_1x_visdrone_2019.py
# python tools/test.py E:\PythonProject\mmdetection\configs\faster_rcnn\faster-rcnn_r50_fpn_ciou_1x_visdrone_2019.py work_dirs/faster-rcnn_r50_fpn_ciou_1x_visdrone_2019/epoch_12.pth
# python tools/test.py configs/faster_rcnn/faster-rcnn_r50_fpn_ciou_1x_visdrone_2019.py work_dirs/faster-rcnn_r50_fpn_ciou_1x_visdrone_2019/epoch_12.pth --show --show-dir test_save
# python tools/test.py E:\PythonProject\mmdetection\configs\faster_rcnn\faster-rcnn_r50_fpn_ciou_1x_visdrone_2019.py work_dirs/faster-rcnn_r50_fpn_ciou_1x_visdrone_2019/epoch_12.pth  --out work_dirs/faster-rcnn_r50_fpn_ciou_1x_visdrone_2019/results.pkl
# python tools/analysis_tools/confusion_matrix.py E:\PythonProject\mmdetection\configs\faster_rcnn\faster-rcnn_r50_fpn_ciou_1x_visdrone_2019.py work_dirs/faster-rcnn_r50_fpn_ciou_1x_visdrone_2019/results.pkl ./work_dirs/cm_output --show
