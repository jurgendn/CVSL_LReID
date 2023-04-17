import os

import numpy as np
import torch
from torch import nn


# Main
class ResNet50_Keypoint_AdaIN_8():

    def __init__(self, cfg):
        self.cfg = cfg
        self.use_cuda = self.cfg.GPU_ID is not None
        self.device = torch.device('cuda:{}'.format(
            self.cfg.GPU_ID[0])) if self.use_cuda else torch.device('cpu')
        self.cpu_device = torch.device('cpu')

        self.MODEL_PATH = os.path.join(self.cfg.ROOT, 'model')
        self.LOG_PATH = os.path.join(self.cfg.ROOT, 'log')
        self.build_model()  # build model
        self.root = self.cfg.ROOT
        self.name = self.cfg.NAME

        print('------------------------ Options -------------------------')
        for k, v in sorted(cfg.items()):
            if not isinstance(v, dict):
                print('%s: %s' % (k, v))
            else:
                print('%s: ' % k)
                for kk, vv in sorted(v.items()):
                    print('    %s: %s' % (kk, vv))
        print('-------------------------- End ----------------------------')

    def train(self, ):
        # save cfg to the disk during training
        self.check_file_exist(self.LOG_PATH)
        self.check_file_exist(self.MODEL_PATH)
        self.check_file_exist(os.path.join(self.MODEL_PATH, self.cfg.NAME))

        file_name = os.path.join(self.MODEL_PATH, self.cfg.NAME, 'opt.txt')
        self.opt_file = open(file_name, 'w')
        self.opt_file.write(
            '------------------------ Options -------------------------\n')
        for k, v in sorted(cfg.items()):
            if not isinstance(v, dict):
                self.opt_file.write('%s: %s \n' % (k, v))
            else:
                self.opt_file.write('%s: \n' % k)
                for kk, vv in sorted(v.items()):
                    self.opt_file.write('    %s: %s \n' % (kk, vv))
        self.opt_file.write(
            '-------------------------- End ----------------------------\n')
        self.opt_file.write(
            '--------------- ResNet50_Keypoint_AdaIN_8 -----------------\n')
        self.opt_file.write(
            '\n------------------------ Accuracy -------------------------\n')

        self.build_optimizer()

        train_data = dataloader.LTCC_DistMap_Loader(root=cfg.TRAIN.ROOT,
                                                    size=cfg.TRAIN.SIZE,
                                                    num_cls=cfg.NUM_CLASS,
                                                    hm_size=[(12, 6)],
                                                    index=cfg.TRAIN.INDEX,
                                                    sigma=2,
                                                    phase='train')
        train_loader = Data.DataLoader(train_data,
                                       batch_size=self.cfg.TRAIN.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=8,
                                       drop_last=True)
        gallery_data = dataloader.LTCC_DistMap_Loader(
            root=self.cfg.TEST.GALLERY,
            size=self.cfg.TRAIN.SIZE,
            num_cls=self.cfg.NUM_CLASS,
            hm_size=[(12, 6)],
            index=None,
            sigma=2,
            phase='eval')
        gallery_loader = Data.DataLoader(gallery_data,
                                         batch_size=self.cfg.TEST.BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=8,
                                         drop_last=False)
        query_data = dataloader.LTCC_DistMap_Loader(root=self.cfg.TEST.QUERY,
                                                    size=self.cfg.TRAIN.SIZE,
                                                    num_cls=self.cfg.NUM_CLASS,
                                                    hm_size=[(12, 6)],
                                                    index=None,
                                                    sigma=2,
                                                    phase='eval')
        query_loader = Data.DataLoader(query_data,
                                       batch_size=self.cfg.TEST.BATCH_SIZE,
                                       shuffle=False,
                                       num_workers=8,
                                       drop_last=False)

        for epoch in range(1, self.cfg.TRAIN.MAX_EPOCH + 1):
            epoch_loss = 0
            self.scheduler.step()
            for step, data in enumerate(train_loader):
                begin = time.time()

                # #############################
                # (1) Data process
                # #############################
                img, _, kp, heatmap, distmap, label, cloth_label, path = data
                img = img.to(self.device)
                kp = kp.float().to(self.device)
                label = label.to(self.device)
                cloth_label = cloth_label.to(self.device)

                # #############################
                # (2) Forward
                # #############################
                global_cls, local_cls, cloth_cls1, cloth_cls2 = self.model(
                    img, kp, feat=False)

                # #############################
                # (3) Loss
                # #############################
                loss1 = self.loss_func(global_cls, label)
                loss2 = self.loss_func(local_cls, label)
                loss3 = self.loss_func(cloth_cls1, cloth_label)
                loss4 = self.loss_func(cloth_cls2, cloth_label)
                loss = loss1 + loss4 * 0.6 + loss2 * 0.5 + loss3 * 0.3

                # #############################
                # (4) Display and Backward
                # #############################
                epoch_loss += loss.item()
                print(
                    'Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  |  CEL_loss1: {:.6f}  |  CEL_loss2: {:.6f}  |  Cloth_loss1: {:.6f}  |  Cloth_loss2: {:.6f}  |  Time: {:.3f}'
                    .format(epoch, self.cfg.TRAIN.MAX_EPOCH, step + 1,
                            len(train_loader),
                            self.optimizer.param_groups[0]['lr'], loss1.item(),
                            loss2.item(), loss3.item(), loss4.item(),
                            time.time() - begin))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % self.cfg.TRAIN.SNAPSHOT == 0:
                # #############################
                # (5) Validate
                # #############################
                self.model.eval()
                with torch.no_grad():
                    gallery_info = self.extract_feature(
                        gallery_loader, 'gallery')
                    query_info = self.extract_feature(query_loader, 'query')
                    cmc, map = self.evaluate2(gallery_info, query_info)

                    print(
                        '-- Epoch:%d, Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'
                        % (epoch, cmc[0], cmc[4], cmc[9], map))
                    self.opt_file.write(
                        'Epoch:%d, Rank@1:%.4f, Rank@5:%.4f, Rank@10:%.4f, Rank@15:%.4f, Rank@20:%.4f, mAP:%.4f \n'
                        %
                        (epoch, cmc[0], cmc[4], cmc[9], cmc[14], cmc[19], map))
                    self.opt_file.flush()

                    self.summary.add_scalar('epoch_rank1', cmc[0], epoch)
                    self.summary.add_scalar('epoch_map', map, epoch)
                    self.summary.add_scalar('epoch_loss',
                                            epoch_loss / len(train_data),
                                            epoch)
                self.model.train()

            # #############################
            # (6) Save
            # #############################
            if epoch % self.cfg.TRAIN.CHECKPOINT == 0:
                self.save_model(epoch, self.MODEL_PATH, self.model, self.name,
                                self.optimizer)

        self.summary.close()
        self.opt_file.close()

    def visualize_ranklist(self, model_path, topk):
        self.model.eval()
        gallery_data = dataloader.LTCC_DistMap_Loader(
            root=self.cfg.TEST.GALLERY,
            size=self.cfg.TRAIN.SIZE,
            num_cls=self.cfg.NUM_CLASS,
            hm_size=[(12, 6)],
            index=['Cloth_Labled_Clean_Data/data_split/changed_test.txt'],
            sigma=2,
            phase='eval')
        gallery_loader = Data.DataLoader(gallery_data,
                                         batch_size=self.cfg.TEST.BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=8,
                                         drop_last=False)
        query_data = dataloader.LTCC_DistMap_Loader(
            root=self.cfg.TEST.QUERY,
            size=self.cfg.TRAIN.SIZE,
            num_cls=self.cfg.NUM_CLASS,
            hm_size=[(12, 6)],
            index=['Cloth_Labled_Clean_Data/data_split/changed_test.txt'],
            sigma=2,
            phase='eval')
        query_loader = Data.DataLoader(query_data,
                                       batch_size=self.cfg.TEST.BATCH_SIZE,
                                       shuffle=False,
                                       num_workers=8,
                                       drop_last=False)

        with torch.no_grad():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)['state_dict'])

            gallery_info = self.extract_feature(gallery_loader, 'gallery')
            query_info = self.extract_feature(query_loader, 'query')

            query_feature = query_info['feature']
            query_cam = np.array(query_info['camera'])
            query_label = np.array(query_info['label'])
            query_cloth = np.array(query_info['cloth'])
            query_name = query_info['name']
            gallery_feature = gallery_info['feature']
            gallery_cam = np.array(gallery_info['camera'])
            gallery_label = np.array(gallery_info['label'])
            gallery_cloth = np.array(gallery_info['cloth'])
            gallery_name = gallery_info['name']

            for i in range(len(query_label)):
                # -----   modify this part of codes for different metrci learning  ------
                # -----   This part also can be replaced by self._evalute_  ------
                qf = query_feature[i]
                ql = query_label[i]
                qc = query_cam[i]
                qcl = query_cloth[i]
                gf = gallery_feature
                gl = gallery_label
                gc = gallery_cam
                gcl = gallery_cloth

                qff = qf.view(-1, 1)
                score = torch.mm(gf, qff)
                score = score.squeeze(1).to(self.cpu_device)
                score = score.numpy()
                index = np.argsort(score)  #from small to large
                index = index[::-1]
                # good index
                query_index = np.argwhere(gl == ql)  # same id
                camera_index = np.argwhere(gc == qc)  # same cam
                cloth_index = np.argwhere(gcl == qcl)  # same id same cloth

                good_index = np.setdiff1d(
                    query_index, camera_index,
                    assume_unique=True)  # same id different cam
                good_index = np.setdiff1d(
                    good_index, cloth_index,
                    assume_unique=True)  # same id different cloth
                junk_index1 = np.argwhere(gl == -1)  # id == -1
                junk_index2 = np.intersect1d(query_index,
                                             camera_index)  # same id same cam
                junk_index2 = np.union1d(junk_index2, cloth_index)
                junk_index = np.append(junk_index2, junk_index1)  #.flatten())

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
                        cv2.rectangle(image, (2, 2), (192, 384),
                                      color=(0, 0, 255),
                                      thickness=5)
                    B, G, R = cv2.split(image)
                    image = cv2.merge([R, G, B])
                    return transforms.ToTensor()(
                        transforms.ToPILImage()(image))

                samples = torch.FloatTensor(topk + 1, 3, 384, 192).fill_(255.)
                samples[0] = read_image(os.path.join(self.cfg.TEST.QUERY,
                                                     query_name[i]),
                                        good=False)
                for k, v in enumerate(index):
                    samples[k + 1] = read_image(path=os.path.join(
                        self.cfg.TEST.GALLERY, gallery_name[v]),
                                                good=v in good_index)
                grid = utils.make_grid(samples,
                                       nrow=11,
                                       padding=30,
                                       normalize=True)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
                    1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)

                # plt.imshow(im)
                # plt.show()
                utils.save_image(samples,
                                 '/home/qxl/ranklist/%s.png' % (query_name[i]),
                                 nrow=9,
                                 padding=30,
                                 normalize=True)

    def extract_feature(self, dataloader, type):
        features = torch.FloatTensor()
        cameras = []
        labels = []
        clothes = []
        names = []
        for data in tqdm.tqdm(dataloader,
                              desc='-- Extract %s features: ' % (type)):
            img, _, kp, heatmap, distmap, _, _, path = data
            label = [self.get_name(p) for p in path]
            camera = [self.get_camera(p) for p in path]
            cloth = [self.get_cloth(p) for p in path]
            name = [p for p in path]
            labels += label
            cameras += camera
            clothes += cloth
            names += name

            n, c, h, w = img.size()
            input_img = img.to(self.device)
            input_kp = kp.float().to(self.device)

            # output1, output2s = self.model(input_img, input_kp, feat=True)
            # ff1, ff2 = output1.data.to(self.cpu_device), output2.data.to(self.cpu_device)
            # ff = torch.cat((ff1, ff2), -1)
            # fnorm = torch.norm(ff, p=2, dim=-1, keepdim=True)
            # ff = ff.div(fnorm.expand_as(ff))

            output1, output2 = self.model(input_img, input_kp, feat=True)
            ff1, ff2 = output1.data.to(self.cpu_device), output2.data.to(
                self.cpu_device)
            fnorm1, fnorm2 = torch.norm(ff1, p=2, dim=1,
                                        keepdim=True), torch.norm(ff2,
                                                                  p=2,
                                                                  dim=1,
                                                                  keepdim=True)
            ff1, ff2 = ff1.div(fnorm1.expand_as(ff1)), ff2.div(
                fnorm2.expand_as(ff2))
            ff = torch.cat((ff1, ff2), -1)

            features = torch.cat((features, ff), 0)
        return {
            'feature': features,
            'camera': cameras,
            'label': labels,
            'cloth': clothes,
            'name': names
        }

    # standard
    def evaluate(self, gallery, query):
        query_feature = query['feature']
        query_cam = np.array(query['camera'])
        query_label = np.array(query['label'])
        gallery_feature = gallery['feature']
        gallery_cam = np.array(gallery['camera'])
        gallery_label = np.array(gallery['label'])

        # query_feature = query_feature.to(self.device)
        # gallery_feature = gallery_feature.to(self.device)

        # print(query_feature.shape)
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = self._evaluate(query_feature[i], query_label[i],
                                             query_cam[i], gallery_feature,
                                             gallery_label, gallery_cam)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC / len(query_label)  #average CMC
        # print(len(CMC))
        # print('-- Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
        return CMC, ap / len(query_label)

    def _evaluate(self, qf, ql, qc, gf, gl, gc):
        query = qf.view(-1, 1)
        # print(query.shape)
        score = torch.mm(gf, query)
        score = score.squeeze(1).to(self.cpu_device)
        score = score.numpy()
        # predict index
        index = np.argsort(score)  #from small to large
        index = index[::-1]
        # index = index[0:2000]
        # good index
        query_index = np.argwhere(gl == ql)
        camera_index = np.argwhere(gc == qc)

        good_index = np.setdiff1d(query_index,
                                  camera_index,
                                  assume_unique=True)
        junk_index1 = np.argwhere(gl == -1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1)  #.flatten())

        CMC_tmp = self.compute_mAP(index, good_index, junk_index)
        return CMC_tmp

    # cloth-changing
    def evaluate2(self, gallery, query):
        query_feature = query['feature']
        query_cam = np.array(query['camera'])
        query_label = np.array(query['label'])
        query_cloth = np.array(query['cloth'])
        gallery_feature = gallery['feature']
        gallery_cam = np.array(gallery['camera'])
        gallery_label = np.array(gallery['label'])
        gallery_cloth = np.array(gallery['cloth'])

        # query_feature = query_feature.to(self.device)
        # gallery_feature = gallery_feature.to(self.device)

        # print(query_feature.shape)
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = self._evaluate2(query_feature[i], query_label[i],
                                              query_cam[i], query_cloth[i],
                                              gallery_feature, gallery_label,
                                              gallery_cam, gallery_cloth)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC / len(query_label)  #average CMC
        # print(len(CMC))
        # print('-- Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
        return CMC, ap / len(query_label)

    def _evaluate2(self, qf, ql, qc, qcl, gf, gl, gc, gcl):
        query = qf.view(-1, 1)
        # print(query.shape)
        score = torch.mm(gf, query)
        score = score.squeeze(1).to(self.cpu_device)
        score = score.numpy()
        # predict index
        index = np.argsort(score)  #from small to large
        index = index[::-1]
        # index = index[0:2000]
        # good index
        query_index = np.argwhere(gl == ql)  # same id
        camera_index = np.argwhere(gc == qc)  # same cam
        cloth_index = np.argwhere(gcl == qcl)  # same id same cloth

        good_index = np.setdiff1d(query_index,
                                  camera_index,
                                  assume_unique=True)  # same id different cam
        good_index = np.setdiff1d(
            good_index, cloth_index,
            assume_unique=True)  # same id different cloth
        junk_index1 = np.argwhere(gl == -1)  # id == -1
        junk_index2 = np.intersect1d(query_index,
                                     camera_index)  # same id same cam
        junk_index2 = np.union1d(junk_index2, cloth_index)
        junk_index = np.append(junk_index2, junk_index1)  #.flatten())

        CMC_tmp = self.compute_mAP(index, good_index, junk_index)
        return CMC_tmp

    def build_model(self, ):
        self.model = network.ResNet50_Keypoint_AdaIN_8(
            num_class=self.cfg.NUM_CLASS, num_cloth=self.cfg.NUM_CLOTH)
        self.model.train()
        if self.use_cuda:
            # self.model.to(self.device)
            self.model.to(self.cfg.GPU_ID[0])
            self.model = torch.nn.DataParallel(self.model, self.cfg.GPU_ID)
        print(self.model)

    def build_optimizer(self, ):
        self.optimizer = torch.optim.SGD(
            [{
                'params': self.model.module.net1.parameters(),
                'lr': self.cfg.TRAIN.LR * 0.1
            }, {
                'params': self.model.module.net2.parameters(),
                'lr': self.cfg.TRAIN.LR * 0.1
            }, {
                'params': self.model.module.embedding.parameters(),
                'lr': self.cfg.TRAIN.LR
            }, {
                'params': self.model.module.AdaIN1.parameters(),
                'lr': self.cfg.TRAIN.LR
            }, {
                'params': self.model.module.AdaIN2.parameters(),
                'lr': self.cfg.TRAIN.LR
            }, {
                'params': self.model.module.cloth_cls1.parameters(),
                'lr': self.cfg.TRAIN.LR
            }, {
                'params': self.model.module.cloth_cls2.parameters(),
                'lr': self.cfg.TRAIN.LR
            }, {
                'params': self.model.module.local_cls.parameters(),
                'lr': self.cfg.TRAIN.LR
            }, {
                'params': self.model.module.global_cls.parameters(),
                'lr': self.cfg.TRAIN.LR
            }],
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True)
        warmup_iter = self.cfg.TRAIN.WARMUP
        gamma = self.cfg.TRAIN.GAMMA
        stepsize = self.cfg.TRAIN.STEPSIZE

        def lr_lambda(epoch):
            # return a multiplier instead of a learning rate
            warmup_factor = 1
            if epoch < warmup_iter:
                alpha = epoch / warmup_iter
                warmup_factor = 0.01 * (1 - alpha) + alpha
            return warmup_factor * gamma**bisect_right(stepsize, epoch)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lr_lambda=lr_lambda)
        # self.scheduler = StepLR(self.optimizer, step_size=self.cfg.TRAIN.STEPSIZE, gamma=0.1)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.summary = SummaryWriter(log_dir='%s/%s' %
                                     (self.LOG_PATH, self.name),
                                     comment='')

    def check_file_exist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_model(self, epoch, path, model, name, optimizer):
        print("-- Saving %d-th epoch model .......... " % (epoch), end='')
        if not os.path.exists(os.path.join(path, name)):
            os.mkdir(os.path.join(path, name))
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            f='%s/%s/%s_%d.pkl' % (path, name, name, epoch))
        print("Finished!")

    def fliplr(self, img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x c x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    @staticmethod
    def get_name(image):
        return image.split('_')[0]

    @staticmethod
    def get_camera(image):
        return int(image.split('_')[2][1:])
        # return int(image.split('_')[1][1])   # for market-1501

    @staticmethod
    def get_cloth(image):
        return image.split('_')[0] + '_' + image.split('_')[1]

    @staticmethod
    def compute_mAP(index, good_index, junk_index):
        ap = 0
        cmc = torch.IntTensor(len(index)).zero_()
        if good_index.size == 0:  # if empty
            cmc[0] = -1
            return ap, cmc

        # remove junk_index
        mask = np.in1d(index, junk_index, invert=True)
        index = index[mask]

        # find good_index index
        ngood = len(good_index)
        mask = np.in1d(index, good_index)
        rows_good = np.argwhere(mask == True)
        rows_good = rows_good.flatten()

        cmc[rows_good[0]:] = 1
        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) * 1.0 / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i * 1.0 / rows_good[i]
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2

        return ap, cmc


# Model
class ResNet50_Keypoint_AdaIN_8(nn.Module):

    def __init__(self, num_class=751, num_cloth=257):
        super(ResNet50_Keypoint_AdaIN_8, self).__init__()
        net = torchvision.models.resnet50(pretrained=True)
        in_features = net.fc.in_features
        # net.layer4[0].downsample[0].stride = (1,1)
        # net.layer4[0].conv2.stride = (1,1)
        self.net1 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                  net.layer1, net.layer2, net.layer3)
        self.net2 = net.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        num_bottleneck = 512
        self.embedding = layers.keypoint_embedding_2(
            bn=True, reduction='max').apply(weights_init_kaiming)
        self.AdaIN2 = layers.AdaIN_SE3(in_channels=2048,
                                       out_channels=2048,
                                       reduction=4).apply(weights_init_kaiming)
        self.AdaIN1 = layers.AdaIN_SE3(in_channels=2048,
                                       out_channels=1024,
                                       reduction=4).apply(weights_init_kaiming)
        self.local_cls = nn.Sequential(
            OrderedDict([
                ('bottleneck',
                 nn.Sequential(
                     OrderedDict([
                         ('fc', nn.Linear(1024, 256)),
                         ('bn', nn.BatchNorm1d(256)),
                         ('relu', nn.ReLU(inplace=True)),
                     ])).apply(weights_init_kaiming)),
                (
                    'classifier',
                    nn.Sequential(
                        OrderedDict([
                            # ('drop', nn.Dropout(p=0.5)),
                            ('cls', nn.Linear(256, num_class)),
                        ])).apply(weights_init_classifier)),
            ]))
        self.global_cls = nn.Sequential(
            OrderedDict([
                ('bottleneck',
                 nn.Sequential(
                     OrderedDict([
                         ('fc', nn.Linear(in_features, num_bottleneck)),
                         ('bn', nn.BatchNorm1d(num_bottleneck)),
                         ('relu', nn.ReLU(inplace=True)),
                     ])).apply(weights_init_kaiming)),
                (
                    'classifier',
                    nn.Sequential(
                        OrderedDict([
                            # ('drop', nn.Dropout(p=0.5)),
                            ('cls', nn.Linear(num_bottleneck, num_class)),
                        ])).apply(weights_init_classifier)),
            ]))
        self.cloth_cls1 = nn.Sequential(
            OrderedDict([
                (
                    'bottleneck',
                    nn.Sequential(
                        OrderedDict([
                            # ('fc', nn.Linear(1024, 256)),
                            # ('bn', nn.BatchNorm1d(256)),
                            # ('relu', nn.ReLU(inplace=True)),
                        ])).apply(weights_init_kaiming)),
                (
                    'classifier',
                    nn.Sequential(
                        OrderedDict([
                            # ('drop', nn.Dropout(p=0.5)),
                            ('cls', nn.Linear(1024, num_cloth)),
                        ])).apply(weights_init_classifier)),
            ]))
        self.cloth_cls2 = nn.Sequential(
            OrderedDict([
                (
                    'bottleneck',
                    nn.Sequential(
                        OrderedDict([
                            # ('fc', nn.Linear(in_features, num_bottleneck)),
                            # ('bn', nn.BatchNorm1d(num_bottleneck)),
                            # ('relu', nn.ReLU(inplace=True)),
                        ])).apply(weights_init_kaiming)),
                (
                    'classifier',
                    nn.Sequential(
                        OrderedDict([
                            # ('drop', nn.Dropout(p=0.5)),
                            ('cls', nn.Linear(2048, num_cloth)),
                        ])).apply(weights_init_classifier)),
            ]))

    def forward(self, x, k, feat=False, tsne=False):
        # x: N x 3 x H x W
        # p: N x C x H x W
        key = self.embedding(k)
        x1 = self.net1(x)
        x1_useful, x1_useless = self.AdaIN1(x1, key)
        x2 = self.net2(x1_useful)
        x2_useful, x2_useless = self.AdaIN2(x2, key)

        x1_useful = self.pool(x1_useful).view(x1_useful.size(0), -1)
        x2_useful = self.pool(x2_useful).view(x2_useful.size(0), -1)
        x1_useless = self.pool(x1_useless).view(x1_useless.size(0), -1)
        x2_useless = self.pool(x2_useless).view(x2_useless.size(0), -1)
        # out_org = self.pool(x2).view(x2.size(0), -1)
        if tsne:
            return self.pool(x1).view(
                x1.size(0), -1), x1_useful, x1_useless, self.pool(x2).view(
                    x2.size(0), -1), x2_useful, x2_useless
            # return torch.mean(x1, dim=1), torch.mean(x1_useful, dim=1), torch.mean(x1_useless, dim=1), \
            #        torch.mean(x2, dim=1), torch.mean(x2_useful, dim=1), torch.mean(x2_useless, dim=1)
        if feat:
            # return x2_useful
            return x1_useful, x2_useful
        else:
            local_cls = self.local_cls(x1_useful)
            global_cls = self.global_cls(x2_useful)
            cloth_cls1 = self.cloth_cls1(x1_useless)
            cloth_cls2 = self.cloth_cls2(x2_useless)

            return global_cls, local_cls, cloth_cls1, cloth_cls2
            # local_fea = self.local_cls.bottleneck(x1_useful)
            # local_cls = self.local_cls.classifier(local_fea)
            # global_fea = self.global_cls.bottleneck(x2_useful)
            # global_cls = self.global_cls.classifier(global_fea)
            # cloth_cls1 = self.cloth_cls1(x1_useless)
            # cloth_cls2 = self.cloth_cls2(x2_useless)
            # return global_cls, local_cls, cloth_cls1, cloth_cls2


# Layers
class keypoint_embedding_2(nn.Module):

    def __init__(self, bn=False, reduction='mean'):
        super(keypoint_embedding_2, self).__init__()
        if bn:
            self.E1 = nn.Sequential(nn.Conv1d(3, 128, 1),
                                    nn.ReLU(inplace=True))
            self.E2 = nn.Sequential(nn.Conv1d(13, 128, 1),
                                    nn.ReLU(inplace=True))
            self.norm = nn.InstanceNorm1d(128)
            self.refine_E = nn.Sequential(nn.Conv1d(128, 256, 1),
                                          nn.ReLU(inplace=True),
                                          nn.InstanceNorm1d(256),
                                          nn.Conv1d(256, 512, 1),
                                          nn.ReLU(inplace=True),
                                          nn.InstanceNorm1d(512),
                                          nn.Conv1d(512, 1024, 1),
                                          nn.ReLU(inplace=True),
                                          nn.InstanceNorm1d(1024))
            self.layer1 = nn.Sequential(nn.Conv2d(1024 * 2, 512, 1),
                                        nn.ReLU(inplace=True),
                                        nn.InstanceNorm2d(512))
            self.layer2 = nn.Sequential(nn.Conv2d(512, 2048, 1),
                                        nn.ReLU(inplace=True),
                                        nn.InstanceNorm2d(2048))
        else:
            self.E1 = nn.Sequential(nn.Conv1d(3, 128, 1),
                                    nn.ReLU(inplace=True))
            self.E2 = nn.Sequential(nn.Conv1d(13, 128, 1),
                                    nn.ReLU(inplace=True))
            self.refine_E = nn.Sequential(nn.Conv1d(128, 256, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv1d(256, 512, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv1d(512, 1024, 1),
                                          nn.ReLU(inplace=True))
            self.layer1 = nn.Sequential(nn.Conv2d(1024 * 2, 512, 1),
                                        nn.ReLU(inplace=True))
            self.layer2 = nn.Sequential(nn.Conv2d(512, 2048, 1))
        self.bn = bn
        self.reduction = reduction
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        # x: B x N x 16
        b, n, _ = x.size()
        coord, pos = x[:, :, :3], x[:, :, 3:]
        coord, pos = coord.transpose(-2,
                                     -1), pos.transpose(-2,
                                                        -1)  # B x C x n_node

        coord = self.E1(coord)  # b x 128 x n_node
        pos = self.E2(pos)  # b x 128 x n_node
        if self.bn:
            emb = self.norm(coord + pos)
        else:
            emb = coord + pos
        emb = self.refine_E(emb)  # b x 1024 x n_node

        emb_i = torch.unsqueeze(emb, 2)  # b x 1024 x 1 x n
        emb_i = emb_i.repeat(1, 1, n, 1)  # b x 1024 x n' x n
        emb_j = torch.unsqueeze(emb, 3)  # b x 1024 x n x 1
        emb_j = emb_j.repeat(1, 1, 1, n)  # b x 1024 x n x n'
        emb = torch.cat((emb_i, emb_j), 1)  # b x 1024*2 x n x n

        emb = self.layer2(self.layer1(emb))  # b x 2048 x n x n
        if self.reduction == 'sum':
            emb = torch.sum(emb, dim=(2, 3))  # b x 2048
        elif self.reduction == 'mean':
            emb = torch.mean(emb, dim=(2, 3))  # b x 2048
        elif self.reduction == 'max':
            # emb = torch.max(emb, dim=(2,3))[0]  # b x 2048
            emb = self.pool(emb).view(b, -1)
        else:
            ValueError('Wrong value of reduction.')

        return emb


class AdaIN_SE3(nn.Module):

    def __init__(self, in_channels, out_channels, reduction):
        super(AdaIN_SE3, self).__init__()
        self.affine_scale = nn.Linear(in_channels, out_channels, bias=True)
        self.affine_bias = nn.Linear(in_channels, out_channels, bias=True)
        self.norm = nn.InstanceNorm2d(out_channels,
                                      affine=False,
                                      momentum=0.0,
                                      track_running_stats=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.SE = nn.Sequential(
            OrderedDict([
                ('fc1',
                 nn.Conv2d(out_channels,
                           out_channels // reduction,
                           kernel_size=1,
                           bias=True,
                           padding=0)),
                ('relu', nn.ReLU(inplace=True)),
                ('fc2',
                 nn.Conv2d(out_channels // reduction,
                           out_channels,
                           kernel_size=1,
                           bias=True,
                           padding=0)),
                ('sigmoid', nn.Sigmoid()),
            ]))
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)).apply(weights_init_kaiming)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)).apply(weights_init_kaiming)

    def forward(self, x, w):
        b, c, _, _ = x.size()
        x_norm = self.norm(x)

        y_scale = 1 + self.affine_scale(w)[:, :, None, None]
        y_bias = 0 + self.affine_bias(w)[:, :, None, None]
        x_scale = (x_norm * y_scale) + y_bias

        x_res = x - x_scale
        x_attn = self.pool(x_res)
        x_attn = self.SE(x_attn)

        x_useful = x_attn * x_res + x_scale
        x_useless = (1. - x_attn) * x_res + x_norm

        x_useful = self.conv1(x_useful)
        x_useless = self.conv2(x_useless)

        return x_useful, x_useless
