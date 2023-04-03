from dynaconf import Dynaconf

GENERAL_CFG = Dynaconf(envvar_prefix="DYNACONF",
                       settings_files=["config/main_cfg.yaml"])

HRNET_CFG = Dynaconf(envvar_prefix="DYNACONF",
                     settings_files=["config/pose_hrnet_w32_256_192.yaml"])
MARKET1501_CFG = Dynaconf(envar_prefix="DYNACONF",
                          settings_file=["config/datasets.yaml"])
BODY_POSE = Dynaconf(envar_prefix="DYNACONF",
                     settings_file=["config/open_pose_body_pose.yaml"])
