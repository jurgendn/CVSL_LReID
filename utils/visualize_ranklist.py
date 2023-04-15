import torch, os
import numpy as np
from src.datasets.base_dataset import TestDataset
from extract_features import extract_feature
import cv2 
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from PIL import Image
from config import get_config
from src.datasets.get_loader import get_query_gallery_loader


conf = get_config(training=False)


def visualize_ranklist(model,  model_path, query_json_path, gallery_json_path, topk):
    model.eval()
    query_loader, gallery_loader = get_query_gallery_loader() 
    with torch.inference_mode():
        model.load_state_dict(torch.load(model_path, map_location=conf.device)['state_dict'])

        gallery_info = extract_feature(gallery_loader, 'gallery')
        query_info = extract_feature(query_loader, 'query')

        query_feature = query_info['feature']
        query_cam = np.array(query_info['camera'])
        query_label = np.array(query_info['label'])
        query_cloth = np.array(query_info['cloth'])
        query_path = query_info['path']
        gallery_feature = gallery_info['feature']
        gallery_cam = np.array(gallery_info['camera'])
        gallery_label = np.array(gallery_info['label'])
        gallery_cloth = np.array(gallery_info['cloth'])
        gallery_path = gallery_info['path']

        for i in range(len(query_label)):
            # -----   modify this part of codes for different metrci learning  ------
            # -----   This part also can be replaced by _evalute_  ------
            qf = query_feature[i]
            ql = query_label[i]
            qc = query_cam[i]
            qcl = query_cloth[i]
            gf = gallery_feature
            gl = gallery_label
            gc = gallery_cam
            gcl = gallery_cloth

            qff = qf.view(-1,1)
            score = torch.mm(gf,qff)
            score = score.squeeze(1).cpu()
            score = score.numpy()
            index = np.argsort(score)  #from small to large
            index = index[::-1]
            # good index
            query_index = np.argwhere(gl==ql) # same id
            camera_index = np.argwhere(gc==qc) # same cam
            cloth_index = np.argwhere(gcl==qcl) # same id same cloth

            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True) # same id different cam
            good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True) # same id different cloth
            junk_index1 = np.argwhere(gl==-1) # id == -1
            junk_index2 = np.intersect1d(query_index, camera_index) # same id same cam
            junk_index2 = np.union1d(junk_index2, cloth_index)
            junk_index = np.append(junk_index2, junk_index1) #.flatten())

            # # good index
            # query_index = np.argwhere(gl==ql)   # same id
            # camera_index = np.argwhere(gc==qc)  # same cam
            #
            # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)  # same id different cam
            # junk_index1 = np.argwhere(gl==-1)    # label == -1, i.e., junk images
            # junk_index2 = np.intersect1d(query_index, camera_index)   #
            # junk_index = np.append(junk_index2, junk_index1) #.flatten())
            # -------------------------------------------------------------------

            mask = np.in1d(index, junk_index, invert=True)
            index = index[mask]
            index = index[:topk]

            def read_image(path, good=False):
                image = cv2.imread(path)
                image = cv2.resize(image, (192, 384))
                if good:
                    cv2.rectangle(image, (2,2), (192, 384), color=(0,0,255), thickness=5)
                B,G,R = cv2.split(image)
                image = cv2.merge([R,G,B])
                return transforms.ToTensor()(transforms.ToPILImage()(image))

            samples = torch.FloatTensor(topk+1, 3, 384, 192).fill_(255.)
            samples[0] = read_image(query_path[i], good=False)
            for k, v in enumerate(index):
                samples[k+1] = read_image(
                    path=os.path.join(gallery_path[v]),
                    good=v in good_index
                )
            grid = make_grid(samples, nrow=11, padding=30, normalize=True)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)

            # plt.imshow(im)
            # plt.show()
            save_image(samples, '/home/qxl/ranklist/%s.png' % (query_path[i].split('/')[-1]), nrow=9, padding=30, normalize=True)
