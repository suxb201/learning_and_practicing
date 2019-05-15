import torch
import torch.nn as nn
import torch.nn.functional as func
import os
import pickle
import numpy as np
import random
from tqdm import tqdm
import time

# basic parameters
TRAIN_TIMES = 100000
BATCH_SIZE = 30
CONTEXT_NUM = 1
# no change parameters
V_FEATURE_DIM = 4096
S_FEATURE_DIM = 4800
N_FEATURE_DIM = 1024  # new feature dim
# which device to use
DEVICE = torch.device('cuda:0')
# dataset path
DATASET = os.path.join('..', 'DataSet', 'TACoS')
TRAIN_VIDEO_DIR = os.path.join(DATASET, 'Interval64_128_256_512_overlap0.8_c3d_fc6')
TRAIN_SENTENCE_FILE = os.path.join(DATASET, 'TACoS', 'train_clip-sentvec.pkl')
TEST_VIDEO_DIR = os.path.join(DATASET, 'Interval128_256_overlap0.8_c3d_fc6')
TEST_SENTENCE_FILE = os.path.join(DATASET, 'TACoS', 'test_clip-sentvec.pkl')
# where to save runtime parameters
RUN_DATA_DIR = "run_data"
RUN_DATA_TRAIN_FILE = os.path.join(RUN_DATA_DIR, 'train_data_')
RUN_DATA_TEST_FILE = os.path.join(RUN_DATA_DIR, 'test_data_')
RUN_DATA_parameter = os.path.join(RUN_DATA_DIR, 'parameter.pkl')


def inter(s0, t0, s1, t1):
    the_inter = min(t0, t1) - max(s0, s1)
    the_inter = max(0, the_inter)
    return the_inter


def union(s0, t0, s1, t1):
    return max(t0, t1) - min(s0, s1)


def iou(s0, t0, s1, t1):
    i = inter(s0, t0, s1, t1)
    u = union(s0, t0, s1, t1)
    return i / u


class TrainDataSet:
    def __init__(self):
        videos = {}
        if os.path.exists(RUN_DATA_TRAIN_FILE + 'videos'):
            videos = pickle.load(open(RUN_DATA_TRAIN_FILE + 'videos', 'rb'))
        else:
            print("loading video feature:")
            for video_name in tqdm(os.listdir(TRAIN_VIDEO_DIR)):
                videos[video_name] = np.load(os.path.join(TRAIN_VIDEO_DIR, video_name))
            pickle.dump(videos, open(RUN_DATA_TRAIN_FILE + 'videos', 'wb'))
        self.videos = videos

        pairs = {}
        if os.path.exists(RUN_DATA_TRAIN_FILE + 'pairs'):
            pairs = pickle.load(open(RUN_DATA_TRAIN_FILE + 'pairs', 'rb'))
        else:
            sentences = {}
            print("loading sentence features:")
            for i in tqdm(pickle.load(open(TRAIN_SENTENCE_FILE, 'rb'), encoding='latin1')):
                name = i[0].split('_')[0]
                if name not in sentences:
                    sentences[name] = []
                    pairs[name] = []
                for j in i[1]:
                    sentences[name].append((i[0], j))
            print("init pairs:")
            for video_name in tqdm(os.listdir(TRAIN_VIDEO_DIR)):
                v_name, v_start, v_end = video_name[:-4].split('_')
                v_start = float(v_start)
                v_end = float(v_end)
                if v_name not in sentences:
                    continue
                for sentence_name, sentences_feature in sentences[v_name]:  # j[0]:name j[1]:feature sentence
                    s_name, s_start, s_end = sentence_name.split('_')
                    assert v_name == s_name
                    s_start = float(s_start)
                    s_end = float(s_end)
                    if iou(s_start, s_end, v_start, v_end) > 0.5 \
                            and inter(s_start, s_end, v_start, v_end) / (v_end - v_start) > 0.85:
                        pairs[v_name].append([video_name, sentences_feature, [s_start - v_start, s_end - v_end]])
            pickle.dump(pairs, open(RUN_DATA_TRAIN_FILE + 'pairs', 'wb'))
        self.pairs = pairs

    def next_batch(self):
        pairs = self.pairs
        batch = random.choices(list(pairs.values()), k=BATCH_SIZE)
        batch = [random.choice(i) for i in batch]
        v_es = []
        s_es = []
        offset_es = []
        for i in batch:
            v_es.append(self.videos[i[0]])
            s_es.append(i[1])
            offset_es.append(i[2])
        return v_es, s_es, offset_es


def to_tensor_on_device(the_list):
    return torch.as_tensor(np.array(the_list, dtype=np.float32)).to(DEVICE)


class TALL(nn.Module):
    def __init__(self):
        super(TALL, self).__init__()
        self.v_fc = nn.Linear(V_FEATURE_DIM, N_FEATURE_DIM).to(DEVICE)
        self.s_fc = nn.Linear(S_FEATURE_DIM, N_FEATURE_DIM).to(DEVICE)
        self.batch_matrix_fc = torch.nn.Linear(N_FEATURE_DIM * 3, 3).to(DEVICE)

    def forward(self, v_es, s_es, offset_es, batch_size):
        v_es = self.v_fc(v_es)
        s_es = self.s_fc(s_es)
        v_es = func.normalize(v_es)
        s_es = func.normalize(s_es)
        # regression matrix
        v_es = torch.unsqueeze(v_es, 1)
        v_es = v_es.m(1, batch_size, 1)
        s_es = torch.unsqueeze(s_es, 0)
        s_es = s_es.repeat(batch_size, 1, 1)
        # matrix
        batch_matrix = torch.cat([v_es * s_es, v_es, s_es], 2)
        # fc
        batch_matrix = self.batch_matrix_fc(batch_matrix)
        # splte
        # print('batch_matrix', batch_matrix.size())
        score_loss, l_loss, r_loss = [torch.squeeze(i, dim=2) for i in torch.split(batch_matrix, 1, 2)]
        # score_loss
        all1 = torch.ones(batch_size, batch_size).to(DEVICE)
        eye = torch.eye(batch_size).to(DEVICE)
        mask_matrix = all1 - eye * 2
        score_loss = torch.log(all1 + torch.exp(score_loss * mask_matrix))
        alpha = (all1 - eye) * (1.0 / batch_size) + eye
        score_loss = score_loss * alpha
        score_loss = torch.mean(score_loss)
        # offset_loss
        # print(l_loss.size())
        l_loss = torch.squeeze(torch.mm(l_loss * eye, torch.ones(batch_size, 1).to(DEVICE)), dim=1)
        r_loss = torch.squeeze(torch.mm(r_loss * eye, torch.ones(batch_size, 1).to(DEVICE)), dim=1)
        # print(l_loss.size())
        # print('')
        offset_loss = torch.stack([l_loss, r_loss], 1)
        offset_loss = torch.abs(offset_loss - offset_es)
        offset_loss = torch.mean(offset_loss)
        # loss
        loss = score_loss + offset_loss * 0.0001
        return batch_matrix, loss, score_loss, offset_loss


def train(model):
    print(model)
    # model.load_state_dict(torch.load(RUN_DATA_parameter))
    train_dataset = TrainDataSet()
    optimizer = torch.optim.Adam(model.parameters())

    loss_all = 0
    for step in range(1, TRAIN_TIMES):
        start_time = time.time()

        v_es, s_es, offset_es = train_dataset.next_batch()
        v_es = to_tensor_on_device(v_es)
        s_es = to_tensor_on_device(s_es)
        offset_es = to_tensor_on_device(offset_es)
        # print(v_es.size())
        # print(s_es.size())
        # print(offset_es.size())
        batch_matrix, loss, score_loss, offset_loss = model(v_es, s_es, offset_es, BATCH_SIZE)
        loss_all += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 4096 == 0:
            torch.save(model.state_dict(), RUN_DATA_parameter)
        if step % 64 == 0:
            print("%7d step : %7.10f %7.3f %7.3f %7.3f %4.3fs" % (step, loss_all / (step + 1), loss, score_loss, offset_loss, time.time() - start_time))
        if step % 50000 == 0:
            test(model)


def calculate_r_at_n_iou(matrix, r_at_n, the_iou_thresh):
    for i in range(r_at_n):
        if iou(matrix[i][1], matrix[i][2], matrix[i][3], matrix[i][4]) > the_iou_thresh:
            return 1
    return 0


def test(model):
    print("start test.")
    iou_thresh = [0.1, 0.2, 0.3, 0.4, 0.5]
    sentences = {}
    videos = {}
    # load sentence
    for i in pickle.load(open(TEST_SENTENCE_FILE, 'rb'), encoding='latin1'):
        name, s_start, s_end = i[0].split('_')
        s_start = float(s_start)
        s_end = float(s_end)
        if name not in sentences:
            sentences[name] = []
        for j in i[1]:
            sentences[name].append((s_start, s_end, j))
    print("test sentences loaded end.")
    # start load video
    if os.path.exists(RUN_DATA_TEST_FILE + 'videos'):
        videos = pickle.load(open(RUN_DATA_TEST_FILE + 'videos', 'rb'))
    else:
        print("no test videos cache, start load videos ...")
        for video_file_name in tqdm(os.listdir(TEST_VIDEO_DIR)):
            v_name, v_start, v_end = video_file_name[:-4].split('_')
            v_start = float(v_start)
            v_end = float(v_end)
            if v_name not in videos:
                videos[v_name] = []
            videos[v_name].append((v_start, v_end, np.load(os.path.join(TEST_VIDEO_DIR, video_file_name))))
        pickle.dump(videos, open(RUN_DATA_TEST_FILE + 'videos', 'wb'))
    print("test video loaded end.")
    # start test
    r_at_1 = [0, 0, 0, 0, 0]
    r_at_5 = [0, 0, 0, 0, 0]
    r_at_10 = [0, 0, 0, 0, 0]
    all_cnt = 0
    for th_movie, movie_name in enumerate(sentences.keys()):
        for th_sentence, sentence in enumerate(sentences[movie_name]):
            # find top-n match video
            result_matrix = []
            for video in videos[movie_name]:
                v_start, v_end, v_data = video
                s_start, s_end, s_data = sentence

                v_es = to_tensor_on_device([v_data])
                s_es = to_tensor_on_device([s_data])

                offset_es = to_tensor_on_device([[s_start - v_start, s_end - v_end]])
                # print(v_es.size())
                # print(s_es.size())
                # print(offset_es.size())
                batch_matrix, _, _, _ = model(v_es, s_es, offset_es, 1)
                score, l_offset, r_offset = torch.squeeze(batch_matrix)
                result_matrix.append((score, v_start + l_offset, v_end + r_offset, s_start, s_end))

            result_matrix.sort(key=lambda x: x[0], reverse=True)
            for i, the_iou_thresh in enumerate(iou_thresh):
                r_at_1[i] += calculate_r_at_n_iou(result_matrix, 1, the_iou_thresh)
                r_at_5[i] += calculate_r_at_n_iou(result_matrix, 5, the_iou_thresh)
                r_at_10[i] += calculate_r_at_n_iou(result_matrix, 10, the_iou_thresh)
            all_cnt += 1
            if all_cnt % 8 == 0:
                print("%7.3f%% %d:" % (all_cnt / 4083 * 100, all_cnt))
                print("          @1       @5      @10")
                for i, the_iou_thresh in enumerate(iou_thresh):
                    print(the_iou_thresh, "%7.3f%% %7.3f%% %7.3f%%" % (r_at_1[i] / all_cnt * 100,
                                                                       r_at_5[i] / all_cnt * 100,
                                                                       r_at_10[i] / all_cnt * 100))
    # output
    print('*****************************')
    print('sentence:', all_cnt)
    print("          @1       @5      @10")
    for i, the_iou_thresh in enumerate(iou_thresh):
        print(the_iou_thresh, "%7.3f%% %7.3f%% %7.3f%%" % (r_at_1[i] / all_cnt * 100, r_at_5[i] / all_cnt * 100, r_at_10[i] / all_cnt * 100))
    print('*****************************')


def main():
    tall = TALL()
    train(tall)


if __name__ == '__main__':
    main()
