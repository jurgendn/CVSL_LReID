import math

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from config import BODY_POSE
from src.models.pose.body_pose import BodyPose
from src.models.pose.utils import draw_bodypose, pad_right_down_corner


def main():
    pose_net = BodyPose(cfg=BODY_POSE)
    with open(BODY_POSE.PRETRAINED, "rb") as f:
        weights = torch.load(f)
    pose_net.load_state_dict(weights)
    pose_net.eval()
    img = Image.open('example/pose_1.jpg')
    oriImg = np.array(img)
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_RGB2BGR)
    # scale_search = [0.5, 1.0, 1.5, 2.0]
    scale_search = BODY_POSE.SCALE_SEARCH
    boxsize = BODY_POSE.BOX_SIZE
    stride = BODY_POSE.STRIDE
    padValue = BODY_POSE.PAD_VALUE
    thre1 = BODY_POSE.THRESHOLD_1
    thre2 = BODY_POSE.THRESHOLD_2
    multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0),
                                 fx=scale,
                                 fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = pad_right_down_corner(
            imageToTest, stride, padValue)
        im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                          (3, 2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        data = torch.from_numpy(im).float()
        with torch.no_grad():
            Mconv7_stage6_L1, Mconv7_stage6_L2 = pose_net(data)
        Mconv7_stage6_L1 = Mconv7_stage6_L1.numpy()
        Mconv7_stage6_L2 = Mconv7_stage6_L2.numpy()

        # extract outputs, resize, and remove padding
        # output 1 is heatmaps
        heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2),
                               (1, 2, 0))  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0),
                             fx=stride,
                             fy=stride,
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] -
                          pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]),
                             interpolation=cv2.INTER_CUBIC)

        # output 0 is PAFs
        paf = np.transpose(np.squeeze(Mconv7_stage6_L1),
                           (1, 2, 0))  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0),
                         fx=stride,
                         fy=stride,
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] -
                  pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

        heatmap_avg += heatmap_avg + heatmap / len(multiplier)
        paf_avg += +paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        one_heatmap = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (one_heatmap >= map_left, one_heatmap >= map_right,
             one_heatmap >= map_up, one_heatmap >= map_down,
             one_heatmap > thre1))
        peaks = list(
            zip(np.nonzero(peaks_binary)[1],
                np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]], ) for x in peaks]
        peak_id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [
            peaks_with_score[i] + (peak_id[i], ) for i in range(len(peak_id))
        ]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # find connection in the specified sequence
    # center 29 is in the position 15
    limbSeq = BODY_POSE.LIMB_SEQ
    # the middle joints heatmap correpondence
    mapIdx = BODY_POSE.MAP_INDEX
    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    norm = max(0.001, norm)
                    vec = np.divide(vec, norm)

                    startend = list(
                        zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                            np.linspace(candA[i][1], candB[j][1],
                                        num=mid_num)))

                    vec_x = np.array([
                        score_mid[int(round(startend[i][1])),
                                  int(round(startend[i][0])), 0]
                        for i in range(len(startend))
                    ])
                    vec_y = np.array([
                        score_mid[int(round(startend[i][1])),
                                  int(round(startend[i][0])), 1]
                        for i in range(len(startend))
                    ])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(
                        vec_y, vec[1])
                    score_with_dist_prior = sum(
                        score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(
                        score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([
                            i, j, score_with_dist_prior,
                            score_with_dist_prior + candA[i][2] + candB[j][2]
                        ])

            connection_candidate = sorted(connection_candidate,
                                          key=lambda x: x[2],
                                          reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack(
                        [connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row
    # is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][
                            indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][indexB] != partBs[i]:
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int),
                                                   2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) +
                                  (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[
                            partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(
                        candidate[connection_all[k][i, :2].astype(int),
                                  2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # subset: n*20 array, 0-17 is the index in candidate
    # 18 is the total score, 19 is the total parts
    # candidate: x, y, score, id

    draw_bodypose(canvas=oriImg, candidate=candidate, subset=subset)


if __name__ == "__main__":
    main()
