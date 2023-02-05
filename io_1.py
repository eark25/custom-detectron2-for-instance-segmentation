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
# djis = ['0243', '0256', '0262', '0269', '0326']
djis = ['0269']
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

# for dji in djis:
#     # input = '/root/mmsegmentation/data/buildingfacade/imgs/cmp_b0022.jpg'
#     # input = '/root/detectron2/20210826_ili_rivervale_mall_-3a.jpg'
#     # input = '/root/detectron2/crack-on-facade-stock-photograph_csp10679891.jpg'
#     # input = '/root/detectron2/output_3/crack-facade-wall-structure-plaster-details-682x1024.jpg'
#     input = '/root/detectron2/DJI_{}.JPG'.format(dji)
#     # input = '/root/detectron2/output_3/patch_7.jpg'
#     # get mask with palette indices (class indices)
#     result = inference_segmentor(model, input)
#     semseg = result[0]
#     # print(semseg)
#     import cv2
#     # get mask with palette color
#     img_with_palette = np.array(palette)[result[0]]
#     cv2.imwrite("semseg.png", img_with_palette[:,:,::-1])
#     # print(img_with_palette)
#     # print(np.unique(img_with_palette))
#     # print(img_with_palette.shape)
#     output = show_result_pyplot(model, input, result, model.PALETTE)
#     # print(output)
#     # print(np.unique(output))
#     # print(output.shape)
#     import cv2
#     # cv2.imwrite('hrnet_imgnet_CLAHE_run/{}_{}_{}_CLAHE.jpg'.format(dji, test_scale, ratios), output)

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
from fpdf import FPDF

for dji in djis:
    pdf = FPDF()
    pdf.add_page()
    effective_page_width = pdf.w - 2 * pdf.l_margin
    pdf.set_font('Times', '', 10)
    # Margin
    m = 10 
    # Page width: Width of A4 is 210mm
    pw = 210 - 2 * m
# for dji in range(1):
#     dji = dji + 3
# for i in range(48):
    # input = '/root/mmsegmentation/data/buildingfacade/imgs/cmp_b0022.jpg'
    # input = '/root/detectron2/20210826_ili_rivervale_mall_-3a.jpg'
    # input = '/root/detectron2/crack-on-facade-stock-photograph_csp10679891.jpg'
    # input = '/root/detectron2/output_3/crack-facade-wall-structure-plaster-details-682x1024.jpg'
    input = '/root/detectron2/DJI_{}.JPG'.format(dji)
    # input = '/root/detectron2/crack_imgs/test/images/DeepCrack_11177.jpg'
    # input = '/root/detectron2/input_patches/0269_patch_{}.jpg'.format(i)
    result = inference_segmentor(model, input)
    semseg = result[0]
    # print(semseg)
    output = show_result_pyplot(model, input, result, model.PALETTE)
    # print(output)
    # print(np.unique(output))
    # print(output.shape)
    import cv2
    cv2.imwrite('hrnet_imgnet_CLAHE_run/{}_{}_{}_CLAHE.jpg'.format(dji, test_scale, ratios), output)
    # get mask with palette color
    img_with_palette = np.array(palette)[result[0]]
    cv2.imwrite("semseg_{}.png".format(dji), img_with_palette[:,:,::-1])
    # print(img_with_palette)
    # print(np.unique(img_with_palette))
    # print(img_with_palette.shape)
    # output = show_result_pyplot(model, input, result, model.PALETTE)
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
    out, instances = v.draw_instance_predictions(outputs["instances_vis"].to("cpu"))
    cv2.imwrite('{}/{}_{}_{}_mt0.4_max1024_nms0.3_no_pca.jpg'.format(cfg.OUTPUT_DIR, dji, cfg.INPUT.MIN_SIZE_TEST, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST), np.array(out.get_image()[:, :, ::-1]))
    # print(outputs) # use these polygons to find skeletons
    # print(instances)
    # print(len(instances))
    # print(polygons[0].reshape(-1, 2))
    out, angles = po.getOutputOrientation(outputs["instances"].pred_masks, np.array(out.get_image()[:, :, ::-1]))

    # apply rules here
    
    cv2.imwrite('{}/{}_{}_{}_mt0.4_max1024_nms0.3.jpg'.format(cfg.OUTPUT_DIR, dji, cfg.INPUT.MIN_SIZE_TEST, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST), out)
    # cv2.imwrite('output_patches/0269_{}_{}_mt0.4_patch_{}.jpg'.format(cfg.INPUT.MIN_SIZE_TEST, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, i), out)

    pdf.image('{}/{}_{}_{}_mt0.4_max1024_nms0.3.jpg'.format(cfg.OUTPUT_DIR, dji, cfg.INPUT.MIN_SIZE_TEST, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST), x = None, y = None, w = effective_page_width, h = 0, type = 'JPEG')
    pdf.ln(8)

    if len(instances) != 0:
        for i, polygons in instances.items():
        # for i, polygon in enumerate(instances):
            print('Processing...')
            print('Crack number {}'.format(i + 1))
            # create palette image from output
            img = np.uint8(img_with_palette[:, :, ::-1])
            # print(img)
            # print(img.shape)
            # create blank image
            bin_img = np.zeros(semseg.shape, dtype=np.uint8)
            # print(bin_img)
            # print(bin_img.shape)
            for polygon in polygons:
                # draw polygon on mask
                polygon = np.int32(polygon.reshape(-1, 2))
                # polygons = np.append(polygons, polygons[0])
                # # polygons = np.array([polygons])
                # polygons = np.int32(polygons.reshape(-1, 2))
                # print('closed polygon: ', polygon)
                # fill polygon in the palette image
                cv2.fillPoly(img, [polygon], (0, 0, 0))
                # fill polygon in the blank image
                cv2.fillPoly(bin_img, [polygon], (255))
            # save palette image with the current instance
            cv2.imwrite("semseg_2_{}_{}.png".format(dji, i + 1), img)
            # save blank image with the current instance
            cv2.imwrite("bin_semseg_2_{}_{}.png".format(dji, i + 1), bin_img)
            # Get the indices of non-zero elements in the image for future usage
            # print(np.nonzero(bin_img))
            area_affected_indices = np.transpose(np.nonzero(bin_img))[:, ::-1]
            # print(area_affected_indices)
            # print(area_affected_indices.shape)
            # draw the affected area on a blank image
            bin_img_affected = np.zeros(semseg.shape, dtype=np.uint8)
            # for point in area_affected_indices:
            #     cv2.circle(bin_img_affected, (int(point[0]), int(point[1])), 3, (255), 2)
            # cv2.imwrite("bin_affected_semseg_2.png", bin_img_affected)
            # number of pixels the crack affected
            crack_area = area_affected_indices.shape[0]
            # get skeleton
            from skimage.morphology import skeletonize
            from skimage.filters import threshold_otsu
            # convert uint8 to binary image
            bin_img = bin_img > threshold_otsu(bin_img)
            # print(bin_img)
            # print(bin_img.shape)
            # convert skeleton image to uint8
            skeleton = np.uint8(skeletonize(bin_img)) * 255
            # print(skeleton)
            # print(skeleton.shape)
            # cv2.imwrite("skel_semseg_2_{}_{}.png".format(dji, i + 1), skeleton)

            # find which component the crack is on
            # fillPoly help getting pixel coordinates covered by a polygon 
            # then we can use those coordinates to find pixel values 
            # Get the pixel values at the specified coordinates
            # polygons = np.int32(polygons)
            # print(semseg)
            # print(area_affected_indices)
            pixel_values = semseg[area_affected_indices[:, 1], area_affected_indices[:, 0]]
            # Print the pixel values
            # print(pixel_values)
            # find the most counted value
            # print(np.bincount(pixel_values).argmax())
            # print("Most frequent value in above array")
            # count each labeled pixel into its own bin
            area_affected_bin = np.bincount(pixel_values, minlength=6)
            # print(area_affected_bin)
            # maximum = max(area_affected_bin)
            # maximum = 19472

            # find class index to get area affected of that class
            max_index = np.argmax(area_affected_bin)
            # max_index = 1

            # count labels in semseg result and find amount of each label store as a dict
            uniques, counts = np.unique(semseg, return_counts=True)
            # print(uniques)
            # print(counts)
            area_affected = dict(zip(uniques, counts))
            # print(area_affected)

            print('The detected crack is on {} component in this image.'.format(classes[max_index] if classes[max_index] != 'background' else 'wall'))

            # find crack density
            # print(crack_area)
            # print(area_affected[max_index])
            crack_density = crack_area / area_affected[max_index] * 100.00
            print(f'Crack density: {crack_density:.2f}%')

            # find crack width
            # find skeleton coordinates
            skel_points = np.transpose(np.nonzero(skeleton))[:, ::-1]
            # print(skel_points)
            # print(skel_points.shape)

            bin_skel = cv2.imread('bin_semseg_2_{}_{}.png'.format(dji, i + 1))
            # cv2.fillPoly(bin_skel, [skel_points], (0))
            # cv2.imwrite("bin_skel_semseg_2.png", bin_skel)

            distances = []
            for point in skel_points:
                cv2.circle(bin_skel, (int(point[0]), int(point[1])), 3, (255), 2)
                # print((int(poFint[0]), int(point[1])))
                # point = tuple(point)
                # Calculate distance of point to the nearest edge of the contour
                distance = cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), True)
                # print(distance)
                distances.append(distance)
            # Calculate the average width of the crack
            crack_width = sum(distances) / len(distances) * 2
            # print(crack_width)
            pixel_per_mm = 0.39
            actual_width = crack_width * pixel_per_mm
            print(f'Crack width: {actual_width:.2f} mm')
            cv2.imwrite("bin_skel_semseg_2_{}_{}.png".format(dji, i + 1), bin_skel)

            current_x = pdf.get_x()
            current_y = pdf.get_y()
            pdf.image("bin_skel_semseg_2_{}_{}.png".format(dji, i + 1), x = None, y = None, w = effective_page_width/3 + 5, h = 0, type = 'PNG')    
            # pdf.ln(8)

            # angle for configuration and position
            theta = angles[i] # example angle in degrees
            print(f'Angle wrt horizontal line: {theta:.2f} degree')
            abs_theta = abs(theta)
            abs_theta = abs_theta % 360
            if (abs_theta >= 337.5 or abs_theta <= 22.5) or (abs_theta >= 157.5 and abs_theta <= 202.5):
                direction = ["horizontal"]
            elif (abs_theta > 22.5 and abs_theta < 45) or (abs_theta > 135 and abs_theta < 157.5) or (abs_theta > 202.5 and abs_theta < 225) or (abs_theta > 315 and abs_theta < 337.5):
                direction = ["diagonal", "horizontal"]
            elif (abs_theta >= 67.5 and abs_theta <= 112.5) or (abs_theta >= 247.5 and abs_theta <= 292.5):
                direction = ["vertical"]
            elif (abs_theta > 45 and abs_theta < 67.5) or (abs_theta > 112.5 and abs_theta < 135) or (abs_theta > 225 and abs_theta < 247.5) or (abs_theta > 292.5 and abs_theta < 315):
                direction = ["diagonal", "vertical"]
            else:
                direction = ["diagonal"]
            print(direction)
            
            # for position
            def find_top_parts(positions):
                # print(positions)
                each_positions = np.bincount(positions, minlength=5)
                # print(each_positions)
                ind = np.argpartition(each_positions, -2)[-2:][::-1]
                # print(ind)
                top_parts = ind[np.argsort(each_positions[ind])][::-1]
                if each_positions[ind[1]] == 0:
                    # print("Index of the top 1 value:", ind[0])
                    top_parts = ind[np.argsort(each_positions[ind[0]])][::-1]
                else:
                    # print("Indices of the top 2 values:", ind)
                    top_parts = ind[np.argsort(each_positions[ind])][::-1]
                # print(top_parts)
                return top_parts

            def find_position(configuration):
                positions = []
                if configuration == 'transverse' or configuration == 'shear':
                    for point in area_affected_indices:
                        x = point[0]
                        if 0 <= x < w//4:
                            positions.append(1)
                        elif w//4 <= x < w//2:
                            positions.append(2)
                        elif w//2 <= x < 3*w//4:
                            positions.append(3)
                        else:
                            positions.append(4)
                    # find most affected parts of the affected component
                    most_affected_parts = find_top_parts(positions)
                    if most_affected_parts[0] == 1:
                        return 'Left one fourth'
                    elif most_affected_parts[0] == 4:
                        return 'Right one fourth'
                    else:
                        return 'Middle fourths'
                elif configuration == 'longitudinal':
                    for point in area_affected_indices:
                        y = point[1]
                        if 0 <= y < h//4:
                            positions.append(1)
                        elif h//4 <= y < h//2:
                            positions.append(2)
                        elif h//2 <= y < 3*h//4:
                            positions.append(3)
                        else:
                            positions.append(4)
                    # find most affected parts of the affected component
                    most_affected_parts = find_top_parts(positions)
                    if (1 in most_affected_parts and 2 in most_affected_parts) or 1 in most_affected_parts:
                        return 'Top fourths'
                    elif (3 in most_affected_parts and 4 in most_affected_parts) or 4 in most_affected_parts:
                        return 'Bottom fourths'
                    else:
                        return 'Middle fourths'

            def beam_severity_index(crack_density, actual_width, configuration, position):
                # Lookup table 1: Beams
                # action = severity_dict[severity_index]
                if crack_density < 1.00:
                    if configuration == 'shear':
                        if position == 'Left one fourth' or position == 'Right one fourth':
                            return 3
                        elif position == 'Middle fourths':
                            return 2
                    elif configuration == 'transverse':
                        if position == 'Left one fourth' or position == 'Right one fourth':
                            return 2
                        elif position == 'Middle fourths':
                            return 3
                    elif configuration == 'longitudinal':
                        if position == 'Middle fourths':
                            return 2
                        elif position == 'Top fourths' or position == 'Bottom fourths':
                            return 3
                elif (crack_density >= 1.00 and crack_density <= 3.00):
                    if configuration == 'shear':
                        if position == 'Left one fourth' or position == 'Right one fourth':
                            return 5
                        elif position == 'Middle fourths':
                            return 4
                    elif configuration == 'transverse':
                        if position == 'Left one fourth' or position == 'Right one fourth':
                            return 4
                        elif position == 'Middle fourths':
                            return 5
                    elif configuration == 'longitudinal':
                        if position == 'Middle fourths':
                            return 4
                        elif position == 'Top fourths' or position == 'Bottom fourths':
                            return 5
                elif crack_density > 3.00:
                    if actual_width <= 5.00:
                        if configuration == 'shear':
                            if position == 'Left one fourth' or position == 'Right one fourth':
                                return 7
                            elif position == 'Middle fourths':
                                return 6
                        elif configuration == 'transverse':
                            if position == 'Left one fourth' or position == 'Right one fourth':
                                return 6
                            elif position == 'Middle fourths':
                                return 7
                        elif configuration == 'longitudinal':
                            if position == 'Middle fourths':
                                return 6
                            elif position == 'Top fourths' or position == 'Bottom fourths':
                                return 7
                    else:
                        return 8

            def column_severity_index(crack_density, actual_width, configuration, position):
                # Lookup table 1: Beams
                # action = severity_dict[severity_index]
                if crack_density < 1.00:
                    if configuration == 'shear':
                        return 2
                    elif configuration == 'transverse':
                        if position == 'Left one fourth' or position == 'Right one fourth':
                            return 3
                        elif position == 'Middle fourths':
                            return 2
                    elif configuration == 'longitudinal':
                        if position == 'Middle fourths':
                            return 2
                        elif position == 'Top fourths' or position == 'Bottom fourths':
                            return 3
                elif (crack_density >= 1.00 and crack_density <= 3.00):
                    if configuration == 'shear':
                        return 4
                    elif configuration == 'transverse':
                        if position == 'Left one fourth' or position == 'Right one fourth':
                            return 5
                        elif position == 'Middle fourths':
                            return 4
                    elif configuration == 'longitudinal':
                        if position == 'Middle fourths':
                            return 4
                        elif position == 'Top fourths' or position == 'Bottom fourths':
                            return 5
                elif crack_density > 3.00:
                    if actual_width <= 5.00:
                        if configuration == 'shear':
                            return 6
                        elif configuration == 'transverse':
                            if position == 'Left one fourth' or position == 'Right one fourth':
                                return 7
                            elif position == 'Middle fourths':
                                return 6
                        elif configuration == 'longitudinal':
                            if position == 'Middle fourths':
                                return 6
                            elif position == 'Top fourths' or position == 'Bottom fourths':
                                return 7
                    else:
                        return 8

            def wall_severity_index(crack_density, actual_width, configuration, position):
                # Lookup table 1: Beams
                # action = severity_dict[severity_index]
                if crack_density < 1.00:
                    if configuration == 'shear':
                        if position == 'Left one fourth' or position == 'Right one fourth':
                            return 2
                        elif position == 'Middle fourths':
                            return 3
                    else:
                        return 2
                elif (crack_density >= 1.00 and crack_density <= 3.00):
                    if configuration == 'shear':
                        if position == 'Left one fourth' or position == 'Right one fourth':
                            return 4
                        elif position == 'Middle fourths':
                            return 5
                    else:
                        return 4
                elif crack_density > 3.00:
                    if actual_width <= 5.00:
                        if configuration == 'shear':
                            if position == 'Left one fourth' or position == 'Right one fourth':
                                return 6
                            elif position == 'Middle fourths':
                                return 7
                        else:
                            return 6
                    else:
                        return 8

            h, w = semseg.shape
            # configuration and position for lookup tables
            # beam
            if classes[max_index] == 'floor':
                if 'diagonal' in direction:
                    configuration = 'shear'
                    position = find_position(configuration)
                    severity_index = beam_severity_index(crack_density, actual_width, configuration, position)
                elif 'vertical' in direction:
                    configuration = 'transverse'
                    position = find_position(configuration)
                    severity_index = beam_severity_index(crack_density, actual_width, configuration, position)
                elif 'horizontal' in direction:
                    configuration = 'longitudinal'
                    position = find_position(configuration)
                    severity_index = beam_severity_index(crack_density, actual_width, configuration, position)

            # column
            elif classes[max_index] == 'column':
                if 'diagonal' in direction:
                    configuration = 'shear'
                    position = find_position(configuration)
                    severity_index = column_severity_index(crack_density, actual_width, configuration, position)
                elif 'vertical' in direction:
                    configuration = 'longitudinal'
                    position = find_position(configuration)
                    severity_index = column_severity_index(crack_density, actual_width, configuration, position)
                elif 'horizontal' in direction:
                    configuration = 'transverse'
                    position = find_position(configuration)
                    severity_index = column_severity_index(crack_density, actual_width, configuration, position)
            
            # wall or undefined classes
            elif classes[max_index] == 'wall' or classes[max_index] == 'background':
                if 'diagonal' in direction:
                    configuration = 'shear'
                    position = find_position(configuration)
                    severity_index = wall_severity_index(crack_density, actual_width, configuration, position)
                elif 'vertical' in direction:
                    configuration = 'transverse'
                    position = find_position(configuration)
                    severity_index = wall_severity_index(crack_density, actual_width, configuration, position)
                elif 'horizontal' in direction:
                    configuration = 'longitudinal'
                    position = find_position(configuration)
                    severity_index = wall_severity_index(crack_density, actual_width, configuration, position)

            else:
                if 'diagonal' in direction:
                    configuration = 'shear'
                    position = find_position(configuration)
                    severity_index = wall_severity_index(crack_density, actual_width, configuration, position)
                elif 'vertical' in direction:
                    configuration = 'transverse'
                    position = find_position(configuration)
                    severity_index = wall_severity_index(crack_density, actual_width, configuration, position)
                elif 'horizontal' in direction:
                    configuration = 'longitudinal'
                    position = find_position(configuration)
                    severity_index = wall_severity_index(crack_density, actual_width, configuration, position)
            
            '''
            keywords:
            middle fourths (middle 2)
            left/right one fourths
            top/bottom fourths (top/bottom 2?)
            any
            '''
            
            # print(h, w)
            # print(centroid)
            # or use crack coordinates of each axis depend on configuration
            # find the part that each coordinate is on and the most two parts 
            # will define which fourths the crack is on
            # also depends on horizontal/vertical
            print('Configuration: {}'.format(configuration))
            print('Position: {} of the {}'.format(position, classes[max_index] if classes[max_index] != 'background' else 'wall'))
                    
            # final results
            severity_dict = {
                1: 'No action needed',
                2: 'General repair but no detailed investigation required',
                3: 'Confirmation by manual visual inspection and general repair',
                4: 'Detailed investigation required',
                5: 'Detailed investigation required and cracks to be sealed if needed',
                6: 'Detailed investigation required and provide immediate protective measures if necessary',
                7: 'Provide imediate protective measures and evacuate if necessary',
                8: 'Evacuate the structure'
            }
            if classes[max_index] == 'floor' or classes[max_index] == 'column' or classes[max_index] == 'wall' or classes[max_index] == 'background':
                print('Severity index for this crack: {}\nSuggested corrective action: {}'.format(severity_index, severity_dict[severity_index]))
            else:
                print('Severity index for this crack: Not in the interested area')
            print('=' * 30)

            # write to pdf
            
            # pdf.cell(w=0, h=5, txt='Crack number {}'.format(i + 1), ln=1)
            # pdf.cell(w=0, h=5, txt='The detected crack is on {} component in this image.'.format(classes[max_index] if classes[max_index] != 'background' else 'wall'), ln=1)
            # pdf.cell(w=0, h=5, txt=f'Crack density: {crack_density:.2f}%', ln=1)
            # pdf.cell(w=0, h=5, txt=f'Crack width: {actual_width:.2f} mm', ln=1)
            # pdf.cell(w=0, h=5, txt=f'Angle wrt horizontal line: {theta:.2f} degree', ln=1)
            # pdf.cell(w=0, h=5, txt='Configuration: {}'.format(configuration), ln=1)
            # pdf.cell(w=0, h=5, txt='Position: {} of the {}'.format(position, classes[max_index] if classes[max_index] != 'background' else 'wall'), ln=1)
            # if classes[max_index] == 'floor' or classes[max_index] == 'column' or classes[max_index] == 'wall' or classes[max_index] == 'background':
            #     pdf.cell(w=0, h=5, txt='Severity index for this crack: {}'.format(severity_index), ln=1)
            #     pdf.cell(w=0, h=5, txt='Suggested corrective action: {}'.format(severity_dict[severity_index]), ln=1)
            
            # else:
            #     pdf.cell(w=0, h=5, txt='Severity index for this crack: Not in the interested area', ln=1)
            
            pdf.set_xy(current_x + effective_page_width*1/3 + 10, current_y)
            if classes[max_index] == 'floor' or classes[max_index] == 'column' or classes[max_index] == 'wall' or classes[max_index] == 'background':
                pdf.multi_cell(w=effective_page_width*2/3 - 15, h=5, txt='Crack number {}'.format(i + 1)
                +'\nThe detected crack is on {} component in this image.'.format(classes[max_index] if classes[max_index] != 'background' else 'wall')
                +f'\nCrack density: {crack_density:.2f}%'
                +f'\nCrack width: {actual_width:.2f} mm'
                +f'\nAngle wrt horizontal line: {theta:.2f} degree'
                +'\nConfiguration: {}'.format(configuration)
                +'\nPosition: {} of the {}'.format(position, classes[max_index] if classes[max_index] != 'background' else 'wall')
                +'\nSeverity index for this crack: {}'.format(severity_index)
                +'\nSuggested corrective action: {}'.format(severity_dict[severity_index]))
                pdf.ln(8)
            else:
                pdf.multi_cell(w=effective_page_width*2/3 - 15, h=5, txt='Crack number {}'.format(i + 1)
                +'\nThe detected crack is on {} component in this image.'.format(classes[max_index] if classes[max_index] != 'background' else 'wall')
                +f'\nCrack density: {crack_density:.2f}%'
                +f'\nCrack width: {actual_width:.2f} mm'
                +f'\nAngle wrt horizontal line: {theta:.2f} degree'
                +'\nConfiguration: {}'.format(configuration)
                +'\nPosition: {} of the {}'.format(position, classes[max_index] if classes[max_index] != 'background' else 'wall')
                +'\nSeverity index for this crack: Not in the interested area')
                pdf.ln(8)

    else:
        print('No crack detected')
        print('=' * 30)
        pdf.multi_cell(w=effective_page_width*2/3 - 15, h=5, txt='No crack detected')
        pdf.ln(8)

    pdf.output(f'./example.pdf', 'F')