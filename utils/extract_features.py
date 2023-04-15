import torch 
from tqdm.auto import tqdm

device = torch.device('cuda')

def extract_feature(model, dataloader, type):
    features = torch.FloatTensor()
    cameras = []
    labels = []
    clothes = []
    paths = []

    for data in tqdm.tqdm(dataloader, desc='-- Extract %s features: ' % (type)):
        imgs, poses, p_ids, cam_ids, cloth_ids, img_paths = data
        
        labels += p_ids
        cameras += cam_ids
        clothes += cloth_ids
        paths += img_paths 

        n, c, h, w = imgs.size()
        input_imgs = imgs.to(device)        
        input_poses = poses.to(device)

        # output1, output2s = model(input_img, input_kp, feat=True)
        # ff1, ff2 = output1.data.to(cpu_device), output2.data.to(cpu_device)
        # ff = torch.cat((ff1, ff2), -1)
        # fnorm = torch.norm(ff, p=2, dim=-1, keepdim=True)
        # ff = ff.div(fnorm.expand_as(ff))

        output = model(input_imgs, input_poses)

        feature = output.data.cpu()
        feature_norm= torch.norm(feature, p=2, dim=1, keepdim=True)
        feature = feature.div(feature_norm.expand_as(feature))

        features = torch.cat((features, feature), 0)
    return {'feature': features,
            'camera': cameras,
            'label': labels,
            'cloth': clothes,
            'path': paths}
