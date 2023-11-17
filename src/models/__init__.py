import torch 
from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet, FTNet_HR, FTNet_Swin, weights_init_classifier
# from src.models.modules.shape_embedding import ShapeEmbedding
from src.models.modules.classifier import Classifier, NormalizedClassifier
from config import BASIC_CONFIG, FT_NET_CFG, SHAPE_EMBEDDING_CFG

conf = BASIC_CONFIG

def build_models(train_shape, class_num, num_clothes, out_features):
    if train_shape: 
            return_f = True 
    else: return_f = False 

    models = {}

    ft_net = FTNet(stride=FT_NET_CFG.R50_STRIDE, class_num=class_num, return_f=return_f)
    
    def load_pretrained_r50(r50_weight_path: str):
        ft_net.load_state_dict(torch.load(f=r50_weight_path), strict=True)

    models['cnn'] = ft_net
    
    # if train_shape:
    #     shape_embedding = ShapeEmbedding(
    #         pose_n_features=SHAPE_EMBEDDING_CFG.POSE_N_FEATURES,
    #         n_hidden=SHAPE_EMBEDDING_CFG.N_HIDDEN,
    #         out_features=SHAPE_EMBEDDING_CFG.OUT_FEATURES,
    #         relation_layers=SHAPE_EMBEDDING_CFG.RELATION_LAYERS)
        
    #     """
    #     need to change the input shape of appearance net and shape net 
    #     if change the relation layers
    #     """
    #     fusion = FusionNet(out_features=out_features)
    #     models['shape'] = shape_embedding
    #     models['fusion'] = fusion

    # Build classifier
    if conf.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        id_classifier = Classifier(feature_dim=out_features, num_classes=class_num)
    else:
        id_classifier = NormalizedClassifier(feature_dim=out_features, num_classes=class_num)

    models['id_clf'] = id_classifier

    if conf.USE_CLOTHES_LOSS:
        clothes_classifier = NormalizedClassifier(feature_dim=out_features, num_classes=num_clothes)
        models['clothes_clf'] = clothes_classifier
    if not conf.TRAIN_FROM_SCRATCH:
        load_pretrained_r50(r50_weight_path=FT_NET_CFG.PRETRAINED)
    
    return models

    