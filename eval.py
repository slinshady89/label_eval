import cv2
import numpy as np
import multiprocessing
from matplotlib import pyplot
from os import getpid
from evaluator import Evaluator
from drawing import draw_bin_coloured
from img_processor import process_image
from diagnostics import calc_best_th
from multiprocessing.managers import BaseManager
from multiprocessing import Pool, TimeoutError
import os
import time

prec_over_rec = np.zeros((256, 3, 2), dtype = np.float)

base_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/08/'
# inf_dir = '20190808_160355/'
# inf_dir = '20190730_170755/'
inf_dir = 'inference/'
label_dir = 'labels_gt/'
img_dir = 'image_2/'

num_img = 3100
lower_bound = 3000

class ImageProcessor(object):

    def process_images(self, th, q):
        # print('evaluating %d images on threshold %d with pid %d' % (num_img - lower_bound, th, getpid()))
        PrecisionOverRecall = np.zeros((num_img, 3, 3), dtype = np.float)
        for k in range(lower_bound, num_img):
            PrecisionOverRecall[k] = process_image(base_dir = base_dir, label_dir = label_dir, inf_dir = inf_dir,
                                                   img_dir = img_dir,
                                                   i = k, threshhold_ = th)
        por = np.zeros((3, 2), dtype = np.float)
        # calculates the mean of Precisions and Recall's over all images for each color channel and returns them without
        # colour quota
        por[0, :] = np.mean(PrecisionOverRecall[lower_bound:num_img, 0], axis = 0)[0:2]
        por[2, :] = np.mean(PrecisionOverRecall[lower_bound:num_img, 2], axis = 0)[0:2]
        # for the green channel the precision and recall is only estimated for images where the green quota is
        # larger than 7% (value estimated by a scatter plot)
        por[1, :] = np.mean(PrecisionOverRecall[PrecisionOverRecall[:, 1, 2] > 0.07, 1], axis = 0)[0:2]
        # print('threshold %d done by pid: %d' % (th, getpid()))
        q.put(por)


def main():

    evaluator = Evaluator(base_dir, img_dir, inf_dir, label_dir, lower_bound, num_img)

    num_proc = 20
    jobs = []

    imgproc = ImageProcessor()
    q = multiprocessing.Queue()
    rets = np.zeros(prec_over_rec.shape)

    for th in range(0, 256):
        p = multiprocessing.Process(target = evaluator.process_images, args = (th, q))
        jobs.append(p)
        p.start()
        # process_images(base_dir, img_dir, inf_dir, label_dir, lower_bound, num_img, th)

    for i, job in enumerate(jobs):
        ret = q.get()
        prec_over_rec[i] = ret

    for job in jobs:
        job.join()

    with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por_green.txt', "w") as f:
        np.savetxt(f, prec_over_rec[:, 1], fmt = '%2.6f')
    with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por_red.txt', "w") as f:
        np.savetxt(f, prec_over_rec[:, 2], fmt = '%2.6f')
    with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por_blue.txt', "w") as f:
        np.savetxt(f, prec_over_rec[:, 0], fmt = '%2.6f')

    f.close()

    with open('/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/Data/por.txt', "r") as f:
        test_read = np.loadtxt(f)

    th_best, por_best = calc_best_th(prec_over_rec)
    print('BEST:')
    print(th_best)
    print(por_best)

    pyplot.plot(prec_over_rec[:, 0, 1], prec_over_rec[:, 0, 0], linestyle = 'None',
                marker = '.', markersize = 1.5, color = 'b')
    pyplot.plot(prec_over_rec[:, 1, 1], prec_over_rec[:, 1, 0], linestyle = 'None',
                marker = '.', markersize = 1.5, color = 'g')
    pyplot.plot(prec_over_rec[:, 2, 1], prec_over_rec[:, 2, 0], linestyle = 'None',
                marker = '.', markersize = 1.5, color = 'r')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.ylim(0, 1)
    pyplot.xlim(0, 1)
    pyplot.grid(b = True)
    pyplot.title('Means of Precision and Recall over all images for thresholds = [0, 255]')

    pyplot.show()


if __name__ == "__main__":
    try:
        print('started')
        t = time.time()
        main()
        e = time.time()
        print('elapsed time %.3f s' % (e - t))
    except KeyboardInterrupt:
        print("Finished")
