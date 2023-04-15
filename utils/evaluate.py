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
            ap_tmp, CMC_tmp = self._evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC/len(query_label) #average CMC
        # print(len(CMC))
        # print('-- Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
        return CMC, ap/len(query_label)

def _evaluate(self,qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).to(self.cpu_device)
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())

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
        ap_tmp, CMC_tmp = self._evaluate2(query_feature[i],query_label[i],query_cam[i],query_cloth[i],
                                            gallery_feature,gallery_label,gallery_cam,gallery_cloth)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    # print(len(CMC))
    # print('-- Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
    return CMC, ap/len(query_label)

def _evaluate2(self,qf,ql,qc,qcl,gf,gl,gc,gcl):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).to(self.cpu_device)
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
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

    CMC_tmp = self.compute_mAP(index, good_index, junk_index)
    return CMC_tmp