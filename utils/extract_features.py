import torch 
from tqdm.auto import tqdm


def extract_feature(dataloader, type):
        features = torch.FloatTensor()
       
        for data in tqdm.tqdm(dataloader, desc='-- Extract %s features: ' % (type)):
            img,  = data
            label = [get_name(p) for p in path]
            camera = [get_camera(p) for p in path]
            cloth = [get_cloth(p) for p in path]
            name = [p for p in path]
            labels += label
            cameras += camera
            clothes += cloth
            names += name

            n, c, h, w = img.size()
            input_img = img.to(device)
            input_kp = kp.float().to(device)

            # output1, output2s = model(input_img, input_kp, feat=True)
            # ff1, ff2 = output1.data.to(cpu_device), output2.data.to(cpu_device)
            # ff = torch.cat((ff1, ff2), -1)
            # fnorm = torch.norm(ff, p=2, dim=-1, keepdim=True)
            # ff = ff.div(fnorm.expand_as(ff))

            output1, output2 = model(input_img, input_kp, feat=True)
            ff1, ff2 = output1.data.to(cpu_device), output2.data.to(cpu_device)
            fnorm1, fnorm2 = torch.norm(ff1, p=2, dim=1, keepdim=True), torch.norm(ff2, p=2, dim=1, keepdim=True)
            ff1, ff2 = ff1.div(fnorm1.expand_as(ff1)), ff2.div(fnorm2.expand_as(ff2))
            ff = torch.cat((ff1, ff2), -1)

            features = torch.cat((features, ff), 0)
        return {'feature': features,
                'camera': cameras,
                'label': labels,
                'cloth': clothes,
                'name': names}
