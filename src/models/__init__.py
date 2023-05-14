import torch 
from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet, FTNet_HR, FTNet_Swin, weights_init_classifier
from src.models.modules.shape_embedding import ShapeEmbedding
from src.models.modules.classifier import Classifier, NormalizedClassifier
from config import BASIC_CONFIG, FT_NET_CFG, SHAPE_EMBEDDING_CFG

def build_models(train_shape, conf, class_num, out_features):
    if train_shape: 
            return_f = True 
    else: return_f = False 

    if conf.USE_RESTNET:
        ft_net = FTNet(stride=FT_NET_CFG.R50_STRIDE, class_num=class_num, return_f=return_f)
    elif conf.USE_HRNET:
        ft_net = FTNet_HR(class_num=class_num, return_f=return_f)
    elif conf.USE_SWIN:
        ft_net = FTNet_Swin(class_num=class_num, return_f=return_f)

    # can try ClassBlock for id_classifier
    if train_shape:
        shape_embedding = ShapeEmbedding(
            pose_n_features=SHAPE_EMBEDDING_CFG.POSE_N_FEATURES,
            n_hidden=SHAPE_EMBEDDING_CFG.N_HIDDEN,
            out_features=SHAPE_EMBEDDING_CFG.OUT_FEATURES,
            relation_layers=SHAPE_EMBEDDING_CFG.RELATION_LAYERS)
        
        """
        need to change the input shape of appearance net and shape net 
        if change the relation layers
        """
        fusion = FusionNet(out_features=out_features)

        id_classifier = Classifier(out_features, class_num)

    if conf.USE_CLOTHES_LOSS:
        pass
    
    if not conf.TRAIN_FROM_SCRATCH:
        load_pretrained_r50(r50_weight_path=FT_NET_CFG.PRETRAINED)

def load_pretrained_r50(self, r50_weight_path: str):
    self.ft_net.load_state_dict(torch.load(f=r50_weight_path), strict=True)