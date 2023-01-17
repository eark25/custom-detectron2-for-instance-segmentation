# building element part
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmseg.models import build_segmentor
import mmcv
from mmcv.runner import load_checkpoint
import numpy as np

config_file = '/root/mmsegmentation/configs/hrnet/myhrnet_imgnet_CLAHE_test.py'
checkpoint_file = '/root/mmsegmentation/hrnet_imgnet_CLAHE_run/best_mIoU_epoch_894.pth'
classes = ('background', 'wall', 'floor', 'column', 'opening', 'facade/deco')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], [0, 11, 123], [118, 20, 12]]
device = 'cuda:3'
djis = ['0269', '0326']
test_scale = 256
thresh = 0.001
ratios = [1.0, 2.0]

img_norm_cfg = dict(
    mean=[255*0.485, 255*0.456, 255*0.406],
    std=[255*0.229, 255*0.224, 255*0.225],
    to_rgb=True
)

crop_size = (512, 512)
inference_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(test_scale, test_scale),
        img_ratios=ratios,
        flip=False,
        transforms=[
            dict(type='CLAHE', clip_limit=3.0, tile_grid_size=(8, 8)),
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# init segmentor
config = mmcv.Config.fromfile(config_file)
config.model.pretrained = None
config.model.train_cfg = None
config.data.test.pipeline = inference_pipeline
# del config.data.test.pipeline[1]
model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
model.CLASSES = checkpoint['meta']['CLASSES']
model.PALETTE = palette
model.cfg = config  # save the config in the model for convenience
model.to(device)
model.eval()

# from patchify import patchify
# import cv2
# full = cv2.imread('/root/detectron2/DJI_0269.JPG')
# patches = patchify(full, (500, 500, 3), step=500)
# print(patches.shape)

# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         patch = patches[i, j, 0]
#         num = i * patches.shape[1] + j
#         cv2.imwrite('input_patches/0269_patch_{}.jpg'.format(num), patch)
#         # patch.save(f"patch_{num}.jpg")
# import sys
# sys.exit(0)

for dji in djis:
    # input = '/root/mmsegmentation/data/buildingfacade/imgs/cmp_b0022.jpg'
    # input = '/root/detectron2/20210826_ili_rivervale_mall_-3a.jpg'
    # input = '/root/detectron2/crack-on-facade-stock-photograph_csp10679891.jpg'
    # input = '/root/detectron2/output_3/crack-facade-wall-structure-plaster-details-682x1024.jpg'
    input = '/root/detectron2/DJI_{}.JPG'.format(dji)
    # input = '/root/detectron2/output_3/patch_7.jpg'
    # get mask with palette indices (class indices)
    result = inference_segmentor(model, input)
    semseg = result[0]
    # print(semseg)
    import cv2
    # get mask with palette color
    img_with_palette = np.array(palette)[result[0]]
    cv2.imwrite("semseg.png", img_with_palette[:,:,::-1])
    # print(img_with_palette)
    # print(np.unique(img_with_palette))
    # print(img_with_palette.shape)
    output = show_result_pyplot(model, input, result, model.PALETTE)
    # print(output)
    # print(np.unique(output))
    # print(output.shape)
    import cv2
    # cv2.imwrite('hrnet_imgnet_CLAHE_run/{}_{}_{}_CLAHE.jpg'.format(dji, test_scale, ratios), output)

# import sys
# sys.exit(0)

# crack detection part
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
import json
import os
import random

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config.config import get_cfg
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import BoxMode
import detectron2.data.transforms as T

import pca_orientation as po

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

for d in ["train"]:
    DatasetCatalog.register("crack_" + d, lambda d=d: get_crack_dicts(d))
    MetadataCatalog.get("crack_" + d).set(thing_classes=["crack"], evaluator_type="coco")
crack_metadata = MetadataCatalog.get("crack_train")
crack_train_dataset = len(DatasetCatalog.get("crack_train"))

epochs = 100
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("crack_train",)
# cfg.DATASETS.TEST = ("crack_test",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = os.path.join('output_clahe_recheck', "model_best.pth")  # path to the model we just trained
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
cfg.MODEL.DEVICE = device
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
# cfg.MODEL.PIXEL_{MEAN/STD}
# cfg.MODEL.PIXEL_MEAN = [128.7035, 125.8532, 120.8661]
# cfg.MODEL.PIXEL_STD = [38.6440, 38.8538, 41.1382]
cfg.INPUT.MIN_SIZE_TRAIN = (256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 928, 960, 992, 1024,)
# # Sample size of smallest side by choice or random selection from range give by
# # INPUT.MIN_SIZE_TRAIN
# cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
# # Maximum size of the side of the image during training
# cfg.INPUT.MAX_SIZE_TRAIN = 1333
# # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 1024
# # Maximum size of the side of the image during testing
# cfg.INPUT.MAX_SIZE_TEST = 1333
cfg.OUTPUT_DIR = 'output_clahe_recheck'

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh  # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3  # if iou > nms_thresh then dont use that box

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

class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.aug = T.AugmentationList([
            ClaheTransform(clip_lim=3.0, win_size=8),
            T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
        ])

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        import torch
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            raw_inputs = T.AugInput(original_image)
            transforms = self.aug(raw_inputs)
            image = raw_inputs.image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

predictor = Predictor(cfg)

from detectron2.utils.visualizer import ColorMode, Visualizer

for dji in djis:
# for i in range(48):
    # input = '/root/mmsegmentation/data/buildingfacade/imgs/cmp_b0022.jpg'
    # input = '/root/detectron2/20210826_ili_rivervale_mall_-3a.jpg'
    # input = '/root/detectron2/crack-on-facade-stock-photograph_csp10679891.jpg'
    # input = '/root/detectron2/output_3/crack-facade-wall-structure-plaster-details-682x1024.jpg'
    input = '/root/detectron2/DJI_{}.JPG'.format(dji)
    # input = '/root/detectron2/input_patches/0269_patch_{}.jpg'.format(i)
    print(input)
    im = cv2.imread(input)
    

    v = Visualizer(im[:, :, ::-1],
                    metadata=crack_metadata, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )

    outputs= predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    # print(outputs["instances_vis"].to("cpu"))
    # use more relaxed mask thresholding prediction for better visualization
    out, polygons = v.draw_instance_predictions(outputs["instances_vis"].to("cpu"))
    # print(polygons) # use these polygons to find skeletons
    # print(polygons[0].reshape(-1, 2))
    out = po.getOutputOrientation(outputs["instances"].pred_masks, np.array(out.get_image()[:, :, ::-1]))

    # apply rules here

    # cv2.imwrite('{}/{}_{}_{}_mt0.4_max1024_nms0.3.jpg'.format(cfg.OUTPUT_DIR, dji, cfg.INPUT.MIN_SIZE_TEST, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST), out)
    # cv2.imwrite('output_patches/0269_{}_{}_mt0.4_patch_{}.jpg'.format(cfg.INPUT.MIN_SIZE_TEST, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, i), out)

# combine part
# output_1 = output
# output_2 = out

# draw polygon on mask
polygons = polygons[0].reshape(-1, 2)
polygons = np.append(polygons, polygons[0])
polygons = np.array([polygons])
polygons = polygons.reshape(-1, 2)
# print(polygons)
img = np.uint8(img_with_palette[:, :, ::-1])
cv2.fillPoly(img, np.int32([polygons]), (0, 0, 0))
cv2.imwrite("semseg_2.png", img)

# find which component the crack is on
# fillPoly help getting pixel coordinates covered by a polygon 
# then we can use those coordinates to find pixel values 
# Get the pixel values at the specified coordinates
polygons = np.int32(polygons)
print(semseg)
print(polygons)
pixel_values = semseg[polygons[:,0], polygons[:,1]]
# Print the pixel values
print(pixel_values)
# find the most counted value
# print(np.bincount(pixel_values).argmax())
print("Most frequent value in above array")
# count each labeled pixel into its own bin
y = np.bincount(pixel_values, minlength=6)
print(y)
maximum = max(y)
  
for i in range(len(y)):
    if y[i] == maximum:
        print(i)