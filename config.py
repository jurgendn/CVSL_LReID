from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from dynaconf import Dynaconf

cfg = Dynaconf(envvar_prefix="DYNACONF",
               settings_files=["config/main_cfg.yaml"])

hrnet_cfg = Dynaconf(envvar_prefix="DYNACONF",
                     settings_files=["config/pose_hrnet_w32_256_192.yaml"])

dataset_cfg = Dynaconf(envar_prefix="DYNACONF",
                          settings_file=["config/datasets.yaml"])
# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.

def get_config(training = True):
    conf = edict()
    conf.input_size = (256, 128)
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """
    input train transform 
    """
    conf.train_transform = None 
    
    conf.test_transform = transforms.Compose([
        transforms.Resize(conf.input_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    """
    input json paths
    """ 
    conf.train_json_path = None 
    conf.query_json_path = None 
    conf.gallery_json_path = None 

    conf.batch_size = 32
    conf.pin_memory = True
    conf.num_workers = 16
    conf.log_path = None 
    conf.save_path = None
    
    """
    input hyper params
    """
    if training:
        conf.lr = None 
     
        