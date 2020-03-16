import getopt
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
import sys; sys.path.insert(1, "/shared/foss-18/detectron2/projects/PointRend")
import point_rend

##########################################################

arguments_strIn = ''
arguments_strOut = ''
arguments_strFrame = ''
arguments_strTreshold = .5

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
	if strOption == '--f' and strArgument != '': arguments_strFrame = strArgument # current frame
	if strOption == '--treshold' and strArgument != '': arguments_strTreshold = strArgument # prediction treshold
# end
im = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)
h, w, c = im.shape
blackframe = np.zeros((h,w,c), np.uint8)
##########################################################

#regular mask_rcnn
#cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = arguments_strTreshold  # set threshold for this model
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#mask_rcnn_predictor = DefaultPredictor(cfg)
#mask_rcnn_outputs = mask_rcnn_predictor(im)

#panoptic segmentation
#cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = arguments_strTreshold  # set threshold for this model
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
#mask_rcnn_panoptic_predictor = DefaultPredictor(cfg)
#mask_rcnn_panoptic_outputs = mask_rcnn_predictor(im)

#PointRend
cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("/shared/foss-18/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
print ("model treshold : ",float(arguments_strTreshold))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(arguments_strTreshold)  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
#once loaded file is here :
#/home/dev18/.torch/fvcore_cache/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
point_rend_preview = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
v = Visualizer(blackframe[:, :, ::-1], coco_metadata, scale=1, instance_mode=ColorMode.SEGMENTATION)
point_rend_mask  = v.draw_instance_mask(outputs["instances"].to("cpu")).get_image()
point_rend_edges = v.draw_instance_edges(outputs["instances"].to("cpu")).get_image()

#output box masks
boxes = outputs["instances"].to("cpu").pred_boxes if outputs["instances"].to("cpu").has("pred_boxes") else None
if boxes is not None:
    boxes = v._convert_boxes(boxes)
    num_instances = len(boxes)
#print (num_instances)
print ("drawing boxes   : ",num_instances," elements")
for i in range(num_instances):
    x0, y0, x1, y1 = boxes[i]
    cv2.rectangle(blackframe,(x0,y0),(x1,y1),(255,255,255),cv2.FILLED)

cv2.imwrite(filename=arguments_strOut+"_preview."+ arguments_strFrame + ".png", img=(point_rend_preview))
cv2.imwrite(filename=arguments_strOut+"_mask."+ arguments_strFrame + ".png", img=(point_rend_mask))
cv2.imwrite(filename=arguments_strOut+"_edges."+ arguments_strFrame + ".png", img=(point_rend_edges))
cv2.imwrite(filename=arguments_strOut+"_box."+ arguments_strFrame + ".png", img=(blackframe))
