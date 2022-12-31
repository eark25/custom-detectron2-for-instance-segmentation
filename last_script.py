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
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import transforms as T

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

class Clahe(T.Augmentation):

    def __init__(self, clip_lim, win_size):
        self.clip_lim = clip_lim
        self.win_size = win_size
        self._init(locals())

    def get_transform(self, image):
        return ClaheTransform(self.clip_lim, self.win_size)
    
class ClaheTransform(T.Transform):

    def __init__(self, clip_lim, win_size):
        super().__init__()
        self.clip_limit = clip_lim
        self.win_size = win_size
        self._set_attributes(locals())

    def apply_image(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.win_size, self.win_size))
        for i in range(img.shape[2]):
            img[:, :, i] = clahe.apply(np.array(img[:, :, i], dtype=np.uint8))
        return img

    def apply_coords(self, coords):
        #coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        #coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation)
        return segmentation

    def inverse(self):
        return ClaheTransform(self.clip_lim, self.win_size)

def build_custom_train_aug(cfg):
    min_size = cfg.INPUT.MIN_SIZE_TRAIN
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    augs = [ClaheTransform(clip_lim=3.0, win_size=8)]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if cfg.INPUT.RANDOM_FLIP != "none":
        augs.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augs

def build_custom_val_aug(cfg):
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    sample_style = "choice"
    augs = [ClaheTransform(clip_lim=3.0, win_size=8)]
    augs.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    return augs

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

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_custom_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, is_train=False, augmentations=build_custom_val_aug(cfg))
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    

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
    json_file = '/root/detectron2/crack_imgs/{}/{}_onlycrack_mt16_deduped.json'.format(img_dir, img_dir)
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

for d in ["train", "val"]:
    DatasetCatalog.register("crack_" + d, lambda d=d: get_crack_dicts(d))
    MetadataCatalog.get("crack_" + d).set(thing_classes=["crack"], evaluator_type="coco")
crack_train_dataset = len(DatasetCatalog.get("crack_train"))
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"], evaluator_type="coco")
# balloon_metadata = MetadataCatalog.get("balloon_train")
# balloon_train_dataset = len(DatasetCatalog.get("balloon_train"))
# evaluator_type = MetadataCatalog.get("balloon_val").evaluator_type
# print(balloon_metadata)
# print(len(balloon_train_dataset))

# To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the training set:
# dataset_dicts = get_balloon_dicts("balloon/train")
# for d in random.sample(dataset_dicts, 1):
#     # print(d["file_name"])
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=1.0)
#     out = visualizer.draw_dataset_dict(d)
#     # cv2.imwrite(d["file_name"].split('/')[2], out.get_image()[:, :, ::-1])

# fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~2 minutes to train 300 iterations on a P100 GPU.
epochs = 400
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("crack_train",)
cfg.DATASETS.TEST = ("crack_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
one_epoch = int(crack_train_dataset / cfg.SOLVER.IMS_PER_BATCH)
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
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
cfg.INPUT.MIN_SIZE_TRAIN = (256, 1024)
# # Sample size of smallest side by choice or random selection from range give by
# # INPUT.MIN_SIZE_TRAIN
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "range"
# # Maximum size of the side of the image during training
# cfg.INPUT.MAX_SIZE_TRAIN = 1333
# # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
cfg.INPUT.MIN_SIZE_TEST = 1024
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
# # Maximum size of the side of the image during testing
# cfg.INPUT.MAX_SIZE_TEST = 1333
cfg.OUTPUT_DIR = 'output_clahe_recheck'
# print(cfg.INPUT.MIN_SIZE_TRAIN)
# print(cfg.INPUT.MAX_SIZE_TRAIN)
# print(cfg.INPUT.MIN_SIZE_TEST)
# print(cfg)
# import sys
# sys.exit(0)

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import wandb
wandb.init(project='Mask-RCNN', resume='allow', anonymous='must', sync_tensorboard=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()