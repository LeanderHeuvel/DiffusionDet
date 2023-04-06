# python3 DiffusionDet/evaluate.py --num-gpus 1 --config-file DiffusionDet/configs/diffdet.coco.res101.yaml --eval-only MODEL.WEIGHTS DiffusionDet/models/diffdet_coco_res101.pth

from dataset import dataset_mapper
from detectron2.evaluation import COCOEvaluator,inference_on_dataset
from detectron2.modeling import build_model

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

#build model
model = build_model(cfg)
model = DiffusionDetWithTTA(cfg, model)

kwargs = may_get_ema_checkpointer(cfg, model)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
#build datasetloader
dataset_name = cfg.DATASETS.TEST[0]

mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
dataloader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
print("created dataloader")
#run through data


def get_all_inputs_outputs():
  for data in dataloader:
    yield data, model(data)



output_folder = cfg.OUTPUT_DIR
evaluator = COCOEvaluator(dataset_name,cfg,True, output_folder, allow_cached_coco=True)
evaluator.reset()

# for inputs, outputs in get_all_inputs_outputs():
#   print("proces batch 1", outputs)
#   evaluator.process(inputs, outputs)
# eval_results = evaluator.evaluate()

with apply_model_ema_and_restore(model):
    results_i = inference_on_dataset(model, dataloader, evaluator)
print(results_i)