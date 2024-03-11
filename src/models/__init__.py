import torch

from configs.factory import (
    FTNetConfig,
    MainConfig,
    MiscellaneusConfig,
    ShapeEmbeddingConfig,
)
from src.models.modules.classifier import Classifier, NormalizedClassifier
from src.models.modules.fusion_net import FusionNet
from src.models.modules.r50 import FTNet
from src.models.modules.shape_embedding import ShapeEmbedding


# def build_models(train_shape, class_num, num_clothes, out_features):
def build_models(
    main_config: MainConfig,
    ftnet_config: FTNetConfig,
    shape_embedding_config: ShapeEmbeddingConfig,
    miscs_config: MiscellaneusConfig,
):
    models = {}

    ft_net = FTNet(config=ftnet_config)

    def load_pretrained_r50(r50_weight_path: str):
        ft_net.load_state_dict(torch.load(f=r50_weight_path), strict=True)

    models["cnn"] = ft_net

    if main_config.train_shape is True:
        shape_embedding = ShapeEmbedding(
            pose_n_features=shape_embedding_config.pose_n_features,
            n_hidden=shape_embedding_config.n_hidden,
            out_features=shape_embedding_config.out_features,
            relation_layers=shape_embedding_config.relation_layers,
        )

        """
        need to change the input shape of appearance net and shape net 
        if change the relation layers
        """
        fusion = FusionNet(config=miscs_config)
        models["shape"] = shape_embedding
        models["fusion"] = fusion

    # Build classifier
    if main_config.cla_loss in ["crossentropy", "crossentropylabelsmooth"]:
        id_classifier = Classifier(
            feature_dim=miscs_config.fusion_net_output_dim,
            num_classes=miscs_config.classifier_num_classes,
        )
    else:
        id_classifier = NormalizedClassifier(
            feature_dim=miscs_config.fusion_net_output_dim,
            num_classes=miscs_config.classifier_num_classes,
        )

    models["id_clf"] = id_classifier

    if main_config.use_clothes_loss is True:
        clothes_classifier = NormalizedClassifier(
            feature_dim=miscs_config.fusion_net_output_dim,
            num_classes=miscs_config.classifier_num_clothes,
        )
        models["clothes_clf"] = clothes_classifier
    if main_config.train_from_scratch is False:
        load_pretrained_r50(r50_weight_path=ftnet_config.pretrained)

    return models
