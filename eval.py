import numpy as np
import multiprocessing
import time
from matplotlib import pyplot
from evaluator import Evaluator
from diagnostics import calc_best_th

prec_over_rec = np.zeros((256, 3, 2), dtype = np.float)

base_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/'
# inf_dir = '20190808_160355/'
# inf_dir = '20190730_170755/'
inf_dir = 'Data/dagger/test_16/'
label_dir = '08/labels_gt/'
img_dir = '08/image_2/'

num_img = 100
lower_bound = 0


import cv2
import os
from diagnostics import preprocess_inference, recall_rgb, precision_rgb
from matplotlib import pyplot as plt

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():

    evaluator = Evaluator(base_dir, img_dir, inf_dir, label_dir, lower_bound)

    eval_list = sorted(os.listdir(base_dir + label_dir))
    threshold = np.array((69, 75, 110), dtype = np.int)
    f1score = np.zeros(shape = (len(eval_list), 4), dtype = np.float)
    for i, names in enumerate(eval_list):
        img = cv2.imread(base_dir + img_dir + eval_list[i])
        inf_label = cv2.imread(base_dir + inf_dir + eval_list[i])
        gt_label = cv2.imread(base_dir + label_dir + eval_list[i])
        if gt_label is None:
            break
        h, w, _ = gt_label.shape
        gt_label = gt_label[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        img = img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
        inf_label = cv2.resize(inf_label, (1024, 256))
        img = cv2.resize(img, (1024, 256))

        inf_label = preprocess_inference(inf_label, threshold)
        # cv2.normalize(img, img, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
        mixed_img = cv2.addWeighted(img, .5, inf_label, .5, 0)
        cv2.imshow('vis', img)
        cv2.waitKey(50)

        prec = precision_rgb(gt_label, inf_label)
        rec = recall_rgb(gt_label, inf_label)
        try:
            f1r = 2.0 * prec[0] * rec[0] / (prec[0] + rec[0])
            f1g = 2.0 * prec[1] * rec[1] / (prec[1] + rec[1])
            f1b = 2.0 * prec[2] * rec[2] / (prec[2] + rec[2])
        except RuntimeWarning as e:
            print(i)
            print(prec, rec)
        f1score[i, :] = [i, f1r, f1g, f1b]

    plt.plot(f1score[:, 0], f1score[:, 1], '-r')
    plt.plot(f1score[:, 0], f1score[:, 2], '-g')
    plt.plot(f1score[:, 0], f1score[:, 3], '-b')
    plt.show()

    if True:
        return

    jobs = []

    q = multiprocessing.Queue()

    num_max_threads = 40
    batch_size = 100
    prq = np.zeros((4000, 3, 3), dtype = np.float)
    for i in range(0, 4000, batch_size):
        p = multiprocessing.Process(target = evaluator.process_batch, args = (q, i, batch_size))
        # if i == 0:
        #     jobs.append([p, i])
        # else:
        jobs.append([p, i])
        p.start()

    ret = None
    for i, job in enumerate(jobs):
        ret = q.get()
        prq[job[1]:job[1] + batch_size] = ret

    print(ret.shape)
    for job in jobs:
        job[0].join()

    # with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por_green.txt', "w") as f:
    #     np.savetxt(f, prec_over_rec[:, 1], fmt = '%2.6f')
    # with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por_red.txt', "w") as f:
    #     np.savetxt(f, prec_over_rec[:, 2], fmt = '%2.6f')
    # with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por_blue.txt', "w") as f:
    #     np.savetxt(f, prec_over_rec[:, 0], fmt = '%2.6f')

    # f.close()

    with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por.txt', "r") as f:
        test_read = np.loadtxt(f)

    # th_best, por_best = calc_best_th(prec_over_rec)
    # print('BEST:')
    # print(th_best)
    # print(por_best)
    #
    # pyplot.plot(prec_over_rec[:, 0, 1], prec_over_rec[:, 0, 0], linestyle = 'None',
    #             marker = '.', markersize = 1.5, color = 'b')
    # pyplot.plot(prec_over_rec[:, 1, 1], prec_over_rec[:, 1, 0], linestyle = 'None',
    #             marker = '.', markersize = 1.5, color = 'g')
    # pyplot.plot(prec_over_rec[:, 2, 1], prec_over_rec[:, 2, 0], linestyle = 'None',
    #             marker = '.', markersize = 1.5, color = 'r')
    # pyplot.xlabel('Recall')
    # pyplot.ylabel('Precision')
    # pyplot.ylim(0, 1)
    # pyplot.xlim(0, 1)
    # pyplot.grid(b = True)
    # pyplot.title('Means of Precision and Recall over all images for thresholds = [0, 255]')
    #
    # pyplot.show()


if __name__ == "__main__":
    print('started')
    t = time.time()
    try:
        main()
        e = time.time()
        print('elapsed time %.3f s' % (e - t))
    except KeyboardInterrupt:
        print("Finished by user")
        e = time.time()
        print('elapsed time %.3f s' % (e - t))

