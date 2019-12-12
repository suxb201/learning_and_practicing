import imageio
import numpy as np
import random

bird_small = imageio.imread("./bird_small.tiff").reshape([-1, 3])
bird_large = imageio.imread("./bird_large.tiff").reshape([-1, 3])

# c = np.random.randint(low=0, high=256, size=[16, 3], dtype=np.int)
c = np.stack([bird_small[i] for i in random.sample(range(0, bird_small.shape[0]), 16)])
loss = -1e9
last_loss = 1e9
iteration = 0
while abs(loss - last_loss) > 10:
    last_loss = loss
    c_sum = np.zeros([16, 3])
    c_cnt = np.zeros([16])
    loss = 0
    for pixel in bird_small:
        res = np.power(pixel - c, 2)
        res = np.sum(res, axis=1)
        index = np.argmin(res)
        min_num = np.min(res)
        assert res[index] == min_num
        c_sum[index] += pixel
        c_cnt[index] += 1
        loss += min_num

    c_sum /= c_cnt[:, None]
    c = c_sum
    iteration += 1
    print(iteration, loss)

    out = bird_large.copy()
    for th, pixel in enumerate(out):
        res = np.power(pixel - c, 2)
        res = np.sum(res, axis=1)
        index = np.argmin(res)
        min_num = np.min(res)
        assert res[index] == min_num
        out[th] = c[index]
    out = out.reshape([538, 538, 3])
    imageio.imwrite("./bird_k_means.tiff", out)
