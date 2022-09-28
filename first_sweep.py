# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

# im = cv2.imread("./input.jpg")

# cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# # print(cfg)
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)

# # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# # print(outputs["instances"].pred_classes)
# # print(outputs["instances"].scores)
# # print(outputs["instances"].pred_boxes)
# # print(outputs["instances"].pred_masks.shape)
# # print(outputs)

# # We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imwrite('output.jpg', out.get_image()[:, :, ::-1])

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

from detectron2.structures import BoxMode

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            # print(anno)
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            # print(anno)
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            # print(px)
            # print(py)
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            # print(poly)
            poly = [p for x in poly for p in x]
            # print(poly)
            # import sys
            # sys.exit(0)

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_crack_dicts(img_dir):
    json_file = '/root/detectron2/crack_imgs/{}/{}_nobg.json'.format(img_dir, img_dir)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    # loop through each image
    for idx, v in enumerate(imgs_anns["images"]): # add [:1] for 1 image
        record = {}
        
        filename = os.path.join('/root/detectron2/crack_imgs/train/images/', v['file_name'])
        height, width = cv2.imread(filename).shape[:2]
        
        # common fields
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        # for obj in [seg for seg in imgs_anns['annotations'] if ((seg["image_id"] == record["image_id"]) and (seg["category_id"] != 0))]:
        for obj in [seg for seg in imgs_anns['annotations'] if seg["image_id"] == record["image_id"]]:
            px = []
            py = []
            obj['bbox_mode'] = BoxMode.XYXY_ABS
            for mask in obj['segmentation']:
                for idx in range(len(mask)):
                    if (idx % 2) == 0:
                        px.append(mask[idx])
                    else:
                        py.append(mask[idx])
                # print('mask', mask)
                # print('px', px)
                # print('py', py)
            obj['bbox'] = [np.min(px), np.min(py), np.max(px), np.max(py)]
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
        # return list[dict]
        return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"], evaluator_type="coco")
balloon_metadata = MetadataCatalog.get("balloon_train")
balloon_train_dataset = len(DatasetCatalog.get("balloon_train"))
# evaluator_type = MetadataCatalog.get("balloon_val").evaluator_type
# print(balloon_metadata)
# print(len(balloon_train_dataset))
# import sys
# sys.exit(0)

# To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the training set:
dataset_dicts = get_balloon_dicts("balloon/train")
for d in random.sample(dataset_dicts, 1):
    # print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=1.0)
    out = visualizer.draw_dataset_dict(d)
    # cv2.imwrite(d["file_name"].split('/')[2], out.get_image()[:, :, ::-1])

# fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~2 minutes to train 300 iterations on a P100 GPU.
epochs = 5
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ( "balloon_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
one_epoch = int(balloon_train_dataset / cfg.SOLVER.IMS_PER_BATCH)
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = int(one_epoch * epochs)   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.WARMUP_ITERS = int(one_epoch)    # warm up iterations before reaching the base learning rate
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.TEST.EVAL_PERIOD = one_epoch
cfg.MODEL.DEVICE = 'cuda:1'
print(cfg)
import sys
sys.exit(0)

import wandb
wandb.init(project='Mask-RCNN', resume='allow', anonymous='must', sync_tensorboard=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

################################################################################################ train_args

import argparse
import warnings
import mmcv
from mmcv import Config

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor

import os.path as osp

import copy

import wandb

'''
learning rate - 1e-4 - 1 ?
batch size - 4/8/16/32
backbone - resnet 18/34/50/101
ignore bg - True/False
crop size - 256/512/1024
keep_ratio resize - True/False
lr_scheduler - poly/none
momentum - 0-1
weight_decay 1e-8/1e-4/1e-2
'''

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--learning_rate', '-lr', dest='learning_rate', metavar='LR', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--batch_size', '-bs', dest='batch_size', metavar='BS', type=int, default=32, help='Batch size')
    parser.add_argument('--backbone', '-bb', dest='backbone', metavar='BB', type=str, default='r50', help='Backbone')
    parser.add_argument('--ignore_bg', '-ib', dest='ignore_bg', metavar='IB', type=bool, default=False, help='Ignore background')
    parser.add_argument('--crop_size', '-cs', dest='crop_size', metavar='CS', type=int, default=512, help='Crop size')
    parser.add_argument('--keep_ratio', '-kr', dest='keep_ratio', metavar='KR', type=bool, default=False, help='Keep ratio')
    parser.add_argument('--scheduler', '-sch', dest='lr_scheduler', metavar='SCH', type=str, default=None, help='Learning rate scheduler')
    parser.add_argument('--momentum', '-mm', dest='momentum', metavar='MM', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', '-wd', dest='weight_decay', metavar='WD', type=float, default=1e-8, help='Weight decay')
    # what is nargs metavar action choices?
    # nargs 
    # - '+'/'*' multiple
    # - '?' use default single
    # metavar displayed name in -h
    # action

    return parser.parse_args()

def main():
    # get args here
    args = get_args()

    # get cfg here
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg = Config.fromfile('../configs/deeplabv3plus/mydeeplabv3plus.py')

    # apply argparsing to cfg here
    if args.learning_rate:
        cfg.optimizer.lr = args.learning_rate
        cfg.lr_config.min_lr = args.learning_rate
    if args.batch_size:
        cfg.optimizer_config.cumulative_iters = int(args.batch_size / cfg.data.samples_per_gpu)
    if args.backbone:
        if args.backbone == 'r50':
            cfg.model.pretrained = 'open-mmlab://resnet50_v1c'
            cfg.model.backbone.depth = 50
        if args.backbone == 'r101':
            cfg.model.pretrained = 'open-mmlab://resnet101_v1c'
            cfg.model.backbone.depth = 101
    if args.ignore_bg:
        cfg.model.decode_head.ignore_index = 0
        cfg.model.decode_head.loss_decode[0].avg_non_ignore = True
        cfg.model.decode_head.loss_decode[1].ignore_index = 0
        cfg.model.auxiliary_head.ignore_index = 0
        cfg.model.auxiliary_head.loss_decode[0].avg_non_ignore = True
        cfg.model.auxiliary_head.loss_decode[1].ignore_index = 0
        cfg.data.train.pipeline[3].ignore_index = 0
        cfg.data.train.pipeline[5].seg_pad_val = 0
        cfg.data.train.pipeline[8].seg_pad_val = 0
        cfg.val_pipeline[3].ignore_index = 0
        cfg.val_pipeline[6].seg_pad_val = 0
        cfg.data.val.type='BuildingFacadeBGDataset'
        cfg.data.val.pipeline[2].transforms[0].ignore_index = 0
        cfg.data.val.pipeline[2].transforms[2].seg_pad_val = 0
        cfg.data.test.type='BuildingFacadeBGDataset'
        cfg.data.test.pipeline[2].transforms[0].ignore_index = 0
        cfg.data.test.pipeline[2].transforms[2].seg_pad_val = 0
    # if args.crop_size:
    #     cfg.crop_size = (args.crop_size, args.crop_size)
    if args.keep_ratio:
        cfg.data.train.pipeline[2].keep_ratio = True
        # cfg.val_pipeline[2].keep_ratio = True
    # if args.lr_scheduler:
    if args.momentum:
        cfg.optimizer.momentum = args.momentum
    if args.weight_decay:
        cfg.optimizer.weight_decay = args.weight_decay

    ######################################################################
    # wandb.init(project='DeepLabv3+', resume='allow', anonymous='must')

    # print(cfg.pretty_text)
    # import sys
    # sys.exit(0)

    # Build the dataset
    # assign dataset catalog
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"], evaluator_type="coco")
    balloon_metadata = MetadataCatalog.get("balloon_train")
    balloon_train_dataset = len(DatasetCatalog.get("balloon_train"))
    # evaluator_type = MetadataCatalog.get("balloon_val").evaluator_type
    # print(balloon_metadata)
    # print(len(balloon_train_dataset))
    # import sys
    # sys.exit(0)

    # To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the training set:
    dataset_dicts = get_balloon_dicts("balloon/train")
    for d in random.sample(dataset_dicts, 1):
        # print(d["file_name"])
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=1.0)
        out = visualizer.draw_dataset_dict(d)
        # cv2.imwrite(d["file_name"].split('/')[2], out.get_image()[:, :, ::-1])

    datasets = [build_dataset(cfg.data.train)]

    # for val loss
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.val_pipeline
        datasets.append(build_dataset(val_dataset))
        # datasets.append(build_dataset(cfg.data.val))

    # Build the detector
    trainer = Trainer(cfg) 

    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    import wandb
    wandb.init(project='Mask-RCNN', resume='allow', anonymous='must', sync_tensorboard=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Train
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())

if __name__ == '__main__':
    main()



