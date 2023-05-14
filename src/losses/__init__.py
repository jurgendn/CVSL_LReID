from torch import nn
from losses.cross_entropy_loss_with_label_smooth import CrossEntropyWithLabelSmooth
from losses.triplet_loss import TripletLoss
from losses.contrastive_loss import ContrastiveLoss
from losses.arcface_loss import ArcFaceLoss
from losses.cosface_loss import CosFaceLoss, PairwiseCosFaceLoss
from losses.circle_loss import CircleLoss, PairwiseCircleLoss
from losses.clothes_based_adversarial_loss import ClothesBasedAdversarialLoss, ClothesBasedAdversarialLossWithMemoryBank
from config import BASIC_CONFIG

config = BASIC_CONFIG

def build_losses(config):
    losses = {}
    # Build identity classification loss
    if config.CLA_LOSS == 'crossentropy':
        criterion_cla = nn.CrossEntropyLoss()
    elif config.CLA_LOSS == 'crossentropylabelsmooth':
        criterion_cla = CrossEntropyWithLabelSmooth()
    elif config.CLA_LOSS == 'arcface':
        criterion_cla = ArcFaceLoss(scale=config.CLA_S, margin=config.CLA_M)
    elif config.CLA_LOSS == 'cosface':
        criterion_cla = CosFaceLoss(scale=config.CLA_S, margin=config.CLA_M)
    elif config.CLA_LOSS == 'circle':
        criterion_cla = CircleLoss(scale=config.CLA_S, margin=config.CLA_M)
    else:
        raise KeyError("Invalid classification loss: '{}'".format(config.CLA_LOSS))

    losses['cla'] = criterion_cla

    if config.USE_PAIRWISE_LOSS:
    # Build pairwise loss
        if config.PAIR_LOSS == 'triplet':
            criterion_pair = TripletLoss(margin=config.PAIR_M)
        elif config.PAIR_LOSS == 'contrastive':
            criterion_pair = ContrastiveLoss(scale=config.PAIR_S)
        elif config.PAIR_LOSS == 'cosface':
            criterion_pair = PairwiseCosFaceLoss(scale=config.PAIR_S, margin=config.PAIR_M)
        elif config.PAIR_LOSS == 'circle':
            criterion_pair = PairwiseCircleLoss(scale=config.PAIR_S, margin=config.PAIR_M)
        else:
            raise KeyError("Invalid pairwise loss: '{}'".format(config.PAIR_LOSS))
        losses['pair'] = criterion_pair
    
    if config.USE_TRIPLET_LOSS:
        if config.TRIPLET_LOSS == 'triplet':
            criterion_triplet = nn.TripletMarginLoss(margin=config.TRIP_M)
        elif config.TRIPLET_LOSS == 'circle':
            pass
        else:
            raise KeyError("Invalid triplet loss")
        losses['triplet'] = criterion_triplet

    if config.USE_CLOTHES_LOSS:
    # Build clothes classification loss
        if config.CLOTHES_CLA_LOSS == 'crossentropy':
            criterion_clothes = nn.CrossEntropyLoss()
        elif config.CLOTHES_CLA_LOSS == 'cosface':
            criterion_clothes = CosFaceLoss(scale=config.CLA_S, margin=0)
        else:
            raise KeyError("Invalid clothes classification loss: '{}'".format(config.CLOTHES_CLA_LOSS))

        # Build clothes-based adversarial loss
        if config.CAL == 'cal':
            criterion_cal = ClothesBasedAdversarialLoss(scale=config.CLA_S, epsilon=config.EPSILON)
        # elif config.CAL == 'calwithmemory':
        #     criterion_cal = ClothesBasedAdversarialLossWithMemoryBank(num_clothes=num_train_clothes, feat_dim=config.MODEL.FEATURE_DIM,
        #                         momentum=config.MOMENTUM, scale=config.CLA_S, epsilon=config.EPSILON)
        else:
            raise KeyError("Invalid clothing classification loss: '{}'".format(config.CAL))
        
        losses['clothes'] = criterion_clothes
        losses['cal'] = criterion_cal


    return losses
