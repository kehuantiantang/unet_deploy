_base_ = [
    './fcn_unet_s5-d16.py', '../pascal_voc12.py',
    '../default_runtime.py', '../schedule_20k.py'
]
model = dict(test_cfg=dict(crop_size=(512, 512), stride=(340, 340)))
# evaluation = dict(metric='mDice')
