
from dynaconf import Dynaconf

cfg = Dynaconf(envvar_prefix="DYNACONF",
               settings_files=["config/main_cfg.yaml"])

hrnet_cfg = Dynaconf(envvar_prefix="DYNACONF",
                     settings_files=["config/pose_hrnet_w32_256_192.yaml"])

dataset = Dynaconf(envar_prefix="DYNACONF",
                          settings_file=["config/datasets.yaml"])
