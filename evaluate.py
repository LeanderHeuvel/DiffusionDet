# python3 DiffusionDet/evaluate.py --num-gpus 1 --config-file DiffusionDet/configs/diffdet.coco.res101.yaml --eval-only MODEL.WEIGHTS DiffusionDet/models/diffdet_coco_res101.pth
import torch
import csv
from dataset import dataset_mapper, coco_dataset_mapper, coco_dataset_mapper_sub
from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.modeling import build_model
from evaluator import CustomEvaluator
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_test_loader

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

from detectron2.checkpoint import DetectionCheckpointer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

##register datasets
DatasetCatalog.register("visdrone_train", dataset_mapper)
MetadataCatalog.get("visdrone_train").thing_classes = ["ignored_regions","pedestrian", "people","bicyle","car","van", "truck","tricycle","awning-tricyle","bus","motor", "others"]
DatasetCatalog.register("visdrone_val", dataset_mapper)
MetadataCatalog.get("visdrone_val").thing_classes = ["ignored_regions","pedestrian", "people","bicyle","car","van", "truck","tricycle","awning-tricyle","bus","motor", "others"]
    
#parse args
args = default_argument_parser().parse_args()
cfg = setup(args)
cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
test_setup = cfg.MODEL.DiffusionDet.NUM_RUNS

## create torch device
device = torch.device(cfg.MODEL.DEVICE)

#register custom dataset
DatasetCatalog.register("coco_val", coco_dataset_mapper)
MetadataCatalog.get("coco_val").thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes

DatasetCatalog.register("coco_val_subset", coco_dataset_mapper_sub)
MetadataCatalog.get("coco_val_subet").thing_classes = MetadataCatalog.get("coco_2017_val").thing_classes

#build model
model = build_model(cfg)

kwargs = may_get_ema_checkpointer(cfg, model)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
#build datasetloader
dataset_name = "coco_val_subset"

mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
#run through data

def results_to_csv(results):
    keys, values = [], []
    for key in results[0].keys():
        keys.append(key)
    for result in results:
      row = []
      for value in result.values():
        if torch.is_tensor(value):
          row.append(value.tolist())
        else:
          row.append(value)
      values.append(row)
      

    with open("frequencies.csv", "w") as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(keys)
        for row in values:
            csvwriter.writerow(row)

def get_all_inputs_outputs(dataloader):
  for data in dataloader:
    yield data, model(data)

output_folder = cfg.OUTPUT_DIR
evaluator = CustomEvaluator()
evaluator.reset()
model.eval()
config = [
  {'threshold':0.0,'dataset_name':"coco_val_subset",'pred_sub':True, "num_proposals":100},
  {'threshold':0.0,'dataset_name':"coco_val_subset",'pred_sub':True, "num_proposals":200},
  {'threshold':0.0,'dataset_name':"coco_val_subset",'pred_sub':True, "num_proposals":400},
  {'threshold':0.0,'dataset_name':"coco_val_subset",'pred_sub':True, "num_proposals":600},
  {'threshold':0.0,'dataset_name':"coco_val_subset",'pred_sub':True, "num_proposals":800},
  {'threshold':0.0,'dataset_name':"coco_val_subset",'pred_sub':True, "num_proposals":1000},
  {'threshold':0.0,'dataset_name':"visdrone_val",'pred_sub':True, "num_proposals":100},
  {'threshold':0.0,'dataset_name':"visdrone_val",'pred_sub':True, "num_proposals":200},
  {'threshold':0.0,'dataset_name':"visdrone_val",'pred_sub':True, "num_proposals":400},
  {'threshold':0.0,'dataset_name':"visdrone_val",'pred_sub':True, "num_proposals":600},
  {'threshold':0.0,'dataset_name':"visdrone_val",'pred_sub':True, "num_proposals":800},
  {'threshold':0.0,'dataset_name':"visdrone_val",'pred_sub':True, "num_proposals":1000}
]
dataset_name=""
results = []
for con in config:
  if dataset_name != con['dataset_name'] or dataloader is None: #prevent creating a new dataloader each time. 
    dataset_name = con['dataset_name']
    dataloader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
  model.dataset_name = con['dataset_name']
  model.set_threshold(con['threshold'])
  model.predict_subset = con['pred_sub']
  model.num_proposals = con['num_proposals']
  with torch.no_grad():
    for inputs, outputs in get_all_inputs_outputs(dataloader):
      evaluator.process(inputs, outputs)
    result = evaluator.evaluate()
    result.update(con)
    results.append(result)
  evaluator.reset()

results_to_csv(results)