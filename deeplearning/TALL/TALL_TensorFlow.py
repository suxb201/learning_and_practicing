import tensorflow as tf
import random
import os
import pickle
import time
import numpy as np
from tqdm import tqdm

# CONST BEGIN
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存

# basic parameters
TRAIN_TIMES = 100000
BATCH_SIZE = 30
CONTEXT_NUM = 1
# no change parameters
V_FEATURE_DIM = 4096
S_FEATURE_DIM = 4800
N_FEATURE_DIM = 1024  # new feature dim

# dataset path
DATASET = os.path.join('..', 'DataSet', 'TACoS')
TRAIN_VIDEO_DIR = os.path.join(DATASET, 'Interval64_128_256_512_overlap0.8_c3d_fc6')
TRAIN_SENTENCE_FILE = os.path.join(DATASET, 'TACoS', 'train_clip-sentvec.pkl')
TEST_VIDEO_DIR = os.path.join(DATASET, 'Interval128_256_overlap0.8_c3d_fc6')
TEST_SENTENCE_FILE = os.path.join(DATASET, 'TACoS', 'test_clip-sentvec.pkl')
# where to save runtime parameters
RUN_DATA_DIR = "run_data"
RUN_DATA_TRAIN_FILE = os.path.join(RUN_DATA_DIR, 'train_data_')
RUN_DATA_TEST_FILE = os.path.join(RUN_DATA_DIR, 'test_data')
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
                        pairs[v_name].append((video_name, sentences_feature, (s_start - v_start, s_end - v_end)))
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


def fc_layer(name, feed, input_dim, output_dim):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w_initializer = tf.constant_initializer(0.)  # tf.random_normal_initializer()
        b_initializer = tf.constant_initializer(0.)
        w = tf.get_variable("w", [input_dim, output_dim], initializer=w_initializer)
        b = tf.get_variable("b", output_dim, initializer=b_initializer)
        fc = tf.nn.xw_plus_b(feed, w, b)
    return fc


class ACRN:
    def __init__(self):
        # self.train_v_es = tf.placeholder(tf.float32, shape=(BATCH_SIZE, CONTEXT_NUM * 2 + 1, V_FEATURE_DIM))
        self.train_v_es = tf.placeholder(tf.float32, shape=(BATCH_SIZE, V_FEATURE_DIM))
        self.train_s_es = tf.placeholder(tf.float32, shape=(BATCH_SIZE, S_FEATURE_DIM))
        self.train_offset_es = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2))

        # self.test_v_es = tf.placeholder(tf.float32, shape=(1, CONTEXT_NUM * 2 + 1, V_FEATURE_DIM))
        self.test_v_es = tf.placeholder(tf.float32, shape=(1, V_FEATURE_DIM))
        self.test_s_es = tf.placeholder(tf.float32, shape=(1, S_FEATURE_DIM))

    def tensor_fusion(self, v_es, s_es, batch_size):
        v_es = fc_layer('v_fc_layer', v_es, V_FEATURE_DIM, output_dim=1024)
        s_es = fc_layer('s_fc_layer', s_es, S_FEATURE_DIM, output_dim=1024)
        v_es = tf.nn.l2_normalize(v_es, dim=1)
        s_es = tf.nn.l2_normalize(s_es, dim=1)

        v_es = tf.reshape(tf.tile(v_es, [batch_size, 1]), [batch_size, batch_size, -1])
        s_es = tf.reshape(tf.tile(s_es, [1, batch_size]), [batch_size, batch_size, -1])

        batch_matrix = tf.reshape(tf.concat([v_es, s_es], 2), [1, batch_size, batch_size, -1])
        batch_matrix = tf.reshape(batch_matrix, [batch_size * batch_size, -1])
        batch_matrix = fc_layer('to_3_fc_layer', batch_matrix, 2048, output_dim=3)
        batch_matrix = tf.reshape(batch_matrix, [batch_size, batch_size, 3])

        return batch_matrix

    def get_loss(self, batch_matrix, offset_es):
        score, l_offset, r_offset = tf.split(batch_matrix, 3, 2)
        l_offset = tf.reshape(l_offset, [BATCH_SIZE, BATCH_SIZE])
        score = tf.reshape(score, [BATCH_SIZE, BATCH_SIZE])
        r_offset = tf.reshape(r_offset, [BATCH_SIZE, BATCH_SIZE])

        eye = tf.eye(BATCH_SIZE)
        all1 = tf.ones([BATCH_SIZE, BATCH_SIZE])

        loss_score = tf.multiply(score, all1 + eye * -2)
        loss_score = tf.exp(loss_score)
        loss_score = tf.add(loss_score, all1)
        loss_score = tf.log(loss_score)
        loss_score = tf.multiply(loss_score, eye + (all1 - eye) / BATCH_SIZE)  # * 0.3 -> / BATCH_SIZE
        loss_score = tf.reduce_mean(loss_score)
        loss_score = loss_score * 100

        l_offset = tf.matmul(tf.multiply(l_offset, eye), tf.ones([BATCH_SIZE, 1]))
        r_offset = tf.matmul(tf.multiply(r_offset, eye), tf.ones([BATCH_SIZE, 1]))
        the_offset = tf.concat((l_offset, r_offset), 1)
        loss_offset = tf.subtract(the_offset, offset_es)
        loss_offset = tf.abs(loss_offset)
        loss_offset = tf.reduce_mean(loss_offset)
        loss = loss_score + loss_offset
        return loss, loss_score, loss_offset

    def loss_back(self, loss):
        optimizer = tf.train.AdamOptimizer(0.9)
        model_end = optimizer.minimize(loss)
        return model_end

    def construct_train_model(self):
        batch_matrix = self.tensor_fusion(self.train_v_es, self.train_s_es, BATCH_SIZE)

        loss, regression_loss, offset_loss = self.get_loss(batch_matrix, self.train_offset_es)
        model_end = self.loss_back(loss)
        return model_end, loss, regression_loss, offset_loss

    def construct_test_model(self):
        batch_matrix_test = self.tensor_fusion(self.test_v_es, self.test_s_es, 1)
        return batch_matrix_test


def calculate_r_at_n_iou(matrix, r_at_n, the_iou_thresh):
    for i in range(r_at_n):
        if iou(matrix[i][1], matrix[i][2], matrix[i][3], matrix[i][4]) > the_iou_thresh:
            return 1
    return 0


def test(acrn, sess, batch_matrix_test):
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
    # start load video
    print("load test video...")
    test_video_time = time.time()
    save_file = os.path.join('run_data_save_suxb201', 'test_set_of_video')
    if os.path.exists(save_file):
        videos = pickle.load(open(save_file, 'rb'))
    else:
        for cnt, i in enumerate(os.listdir(TEST_VIDEO_DIR)):
            if cnt % 32 == 0:
                print('loading test video %2.3f%%' % (cnt / 62741 * 100))
            v_name, v_start, v_end = i[:-4].split('_')
            v_start = float(v_start)
            v_end = float(v_end)
            if v_name not in videos:
                videos[v_name] = []
            videos[v_name].append((v_start, v_end, np.load(os.path.join(TEST_VIDEO_DIR, i))))
        pickle.dump(videos, open(save_file, 'wb'))
    print("test video loaded (use %3.3f s)" % (time.time() - test_video_time))
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
                feed_dict = {
                    acrn.test_v_es: [v_data],
                    acrn.test_s_es: [s_data]
                }
                # start_train = time.time()
                score, l_offset, r_offset = sess.run(batch_matrix_test, feed_dict=feed_dict)[0][0]
                l_offset, r_offset = 0, 0
                # print('test train use', time.time() - start_train)

                # print(l_offset, r_offset)
                # score, l_offset, r_offset = batch_matrix
                result_matrix.append((score, v_start + l_offset, v_end + r_offset, s_start, s_end))  # score get_l get_r real_l real_r

            result_matrix.sort(key=lambda x: x[0], reverse=True)
            # print(result_matrix[0][1], result_matrix[0][2], result_matrix[0][3], result_matrix[0][4])
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


def train(train_set):
    acrn = ACRN()
    model_list = acrn.construct_train_model()
    batch_matrix_test = acrn.construct_test_model()

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    # saver.restore(sess, "./run_data_save_suxb201/TensorFlow_parameter")

    print("variables:")
    for i in tf.trainable_variables():
        print(i)
    print("start train...")

    loss_all = 0
    for step in range(TRAIN_TIMES):

        start = time.time()
        v_es, s_es, offset_es = train_set.next_batch()
        feed_dict = {
            acrn.train_v_es: v_es,
            acrn.train_s_es: s_es,
            acrn.train_offset_es: offset_es
        }

        model_end, loss, loss_score, loss_offset = sess.run(model_list, feed_dict=feed_dict)
        loss_all += loss
        if step % 1000 == 0:
            print("%7d step : %7.10f %7.3f %7.3f %7.3f %4.3fs" % (step, loss_all / (step + 1), loss, loss_score, loss_offset, time.time() - start))
        if (step + 2) % 12800 == 0:
            saver.save(sess, './run_data_save_suxb201/TensorFlow_parameter')
        if (step + 1) % 250000 == 0:
            test(acrn, sess, batch_matrix_test)


def main():
    train_set = TrainDataSet()
    train(train_set)


if __name__ == '__main__':
    main()
