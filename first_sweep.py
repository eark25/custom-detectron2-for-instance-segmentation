# Some basic setup:
import argparse
import warnings
from shapely.errors import ShapelyDeprecationWarning

import wandb

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
from detectron2.structures import BoxMode

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

def get_crack_dicts(img_dir):
    json_file = '/root/detectron2/crack_imgs/{}/{}_onlycrack_mt16.json'.format(img_dir, img_dir)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    # loop through each image
    # for idx, v in enumerate(imgs_anns["images"]):
    #     print(idx, v)
    # import sys
    # sys.exit(0)
    for idx, v in enumerate(imgs_anns["images"]): # add [:1] for 1 image
        record = {}
        
        if 'noncrack' in v['file_name']:
            filename = os.path.join('/root/detectron2/crack_imgs/{}/images/'.format(img_dir), v['file_name'] + '.jpg')
        else:
            filename = os.path.join('/root/detectron2/crack_imgs/{}/images/'.format(img_dir), v['file_name'])
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

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--learning_rate', '-lr', dest='learning_rate', metavar='LR', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', '-bs', dest='batch_size', metavar='BS', type=int, default=8, help='Batch size')
    parser.add_argument('--backbone', '-bb', dest='backbone', metavar='BB', type=str, default='r50', help='Backbone')
    parser.add_argument('--max_train_size', '-mts', dest='max_train_size', metavar='MTS', type=int, default=1024, help='Max train size')
    parser.add_argument('--test_size', '-ts', dest='test_size', metavar='TS', type=int, default=512, help='Test size')
    parser.add_argument('--weight_decay', '-wd', dest='weight_decay', metavar='WD', type=float, default=1e-4, help='Weight decay')

    return parser.parse_args()

def main():
    # register dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("crack_" + d, lambda d=d: get_crack_dicts(d))
        MetadataCatalog.get("crack_" + d).set(thing_classes=["crack"], evaluator_type="coco")
    crack_train_dataset = len(DatasetCatalog.get("crack_train"))

    # get args here
    args = get_args()

    # get cfg here
    cfg = get_cfg()

    # apply argparsing to cfg here
    # COCO Instance Segmentation Baselines with Mask R-CNN
    if args.backbone == 'r50':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    elif args.backbone == 'r101':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    # New baselines using Large-Scale Jitter and Longer Training Schedule
    # elif args.backbone == 'new_r50':
    #     cfg.merge_from_file(model_zoo.get_config_file("new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ.py"))
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ.py")  # Let training initialize from model zoo
    # elif args.backbone == 'new_r101':
    #     cfg.merge_from_file(model_zoo.get_config_file("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"))
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py")  # Let training initialize from model zoo

    # fine-tune a COCO-pretrained Rxxx-FPN Mask R-CNN model on the crack dataset.
    epochs = 80
    cfg.DATASETS.TRAIN = ("crack_train",)
    cfg.DATASETS.TEST = ("crack_val",)
    cfg.DATALOADER.NUM_WORKERS = 0

    if args.batch_size:
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size  # This is the real "batch size" commonly known to deep learning people

    one_epoch = int(crack_train_dataset / cfg.SOLVER.IMS_PER_BATCH)

    if args.learning_rate:
        cfg.SOLVER.BASE_LR = args.learning_rate  # pick a good LR

    cfg.SOLVER.MAX_ITER = int(one_epoch * epochs)   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.WARMUP_ITERS = int(one_epoch)    # warm up iterations before reaching the base learning rate
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (balloon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.TEST.EVAL_PERIOD = one_epoch
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER + 1
    cfg.MODEL.DEVICE = 'cuda:2'
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.SIZE = [0.8, 0.8]
    cfg.INPUT.CROP.TYPE = "relative_range"
    # cfg.MODEL.PIXEL_{MEAN/STD}
    # cfg.MODEL.PIXEL_MEAN = [128.7035, 125.8532, 120.8661]
    # cfg.MODEL.PIXEL_STD = [38.6440, 38.8538, 41.1382]
    
    if args.max_train_size:
        cfg.INPUT.MIN_SIZE_TRAIN = (256, args.max_train_size)
    # # Sample size of smallest side by choice or random selection from range give by
    # # INPUT.MIN_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "range"

    # # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    if args.test_size:
        cfg.INPUT.MIN_SIZE_TEST = args.test_size

    if args.weight_decay:
        cfg.SOLVER.WEIGHT_DECAY = args.weight_decay

    cfg.OUTPUT_DIR = 'output_sweep'

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

    wandb.init(project='mask_rcnn_sweep', resume='allow', anonymous='must', sync_tensorboard=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()



