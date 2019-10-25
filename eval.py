import numpy as np
import multiprocessing
import time
from matplotlib import pyplot
from evaluator import Evaluator
import tikzplotlib as tikz
import json

prec_over_rec = np.zeros((256, 3, 2), dtype = np.float)

base_dir = '/media/localadmin/Test/11Nils/kitti/dataset/sequences/'
# base_dir = '/media/localadmin/Stick_Nils/indice_pooling/'
# inf_dir = '20190808_160355/'
# inf_dir = '20190730_170755/'
# inf_dir = '08/pooling_test/SegNet/'
seq = '%02d' % 9
tested_name = '_dag_' +seq
inf_dir = 'Data/dagger/%s/inf_08/' % seq
plotpng = '.png'
plottex = '.tex'
# inf_dir = '08/20190730_170755/'
# plotpng = '_small.png'
# plottex = '_small.tex'
label_dir = '08/labels/'
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
    # evaluator = Evaluator(base_dir, img_dir, inf_dir, label_dir, lower_bound)

    # eval_list = sorted(os.listdir(base_dir + label_dir))
    # threshold = np.array((69, 75, 110), dtype = np.int)
    # f1score = np.zeros(shape = (len(eval_list), 4), dtype = np.float)
    # for i, names in enumerate(eval_list):
    #     img = cv2.imread(base_dir + img_dir + eval_list[i])
    #     inf_label = cv2.imread(base_dir + inf_dir + eval_list[i])
    #     gt_label = cv2.imread(base_dir + label_dir + eval_list[i])
    #     if gt_label is None:
    #         break
    #     h, w, _ = gt_label.shape
    #     gt_label = gt_label[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
    #     img = img[(h - 256):h, ((w - 1024) // 2): (w - (w - 1024) // 2)]
    #     inf_label = cv2.resize(inf_label, (1024, 256))
    #     img = cv2.resize(img, (1024, 256))
    #
    #     inf_label = preprocess_inference(inf_label, threshold)
    #     # cv2.normalize(img, img, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    #     mixed_img = cv2.addWeighted(img, .5, inf_label, .5, 0)
    #     cv2.imshow('vis', img)
    #     cv2.waitKey(50)
    #
    #     prec = precision_rgb(gt_label, inf_label)
    #     rec = recall_rgb(gt_label, inf_label)
    #     try:
    #         f1r = 2.0 * prec[0] * rec[0] / (prec[0] + rec[0])
    #         f1g = 2.0 * prec[1] * rec[1] / (prec[1] + rec[1])
    #         f1b = 2.0 * prec[2] * rec[2] / (prec[2] + rec[2])
    #     except RuntimeWarning as e:
    #         print(i)
    #         print(prec, rec)
    #     f1score[i, :] = [i, f1r, f1g, f1b]
    #
    # plt.plot(f1score[:, 0], f1score[:, 1], '-r')
    # plt.plot(f1score[:, 0], f1score[:, 2], '-g')
    # plt.plot(f1score[:, 0], f1score[:, 3], '-b')
    # plt.show()
    #
    # if True:
    #     return

    eval_list = sorted(os.listdir(base_dir + label_dir))
    print('len eval list %d' % len(eval_list))
    evaluator = Evaluator(_base_dir = base_dir,
                          _img_dir = img_dir,
                          _inf_label_dir = inf_dir,
                          _gt_label_dir = label_dir,
                          _eval_list = eval_list)

    jobs = []

    q = multiprocessing.Queue()

    batch_size = 50
    prq = np.zeros((4050, 3, 3), dtype = np.float)
    for i in range(0, 4050, batch_size):
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

    for job in jobs:
        job[0].join()


    f1 = np.zeros((4050, 3), dtype = np.float)
    prq2 = []
    for i, item in enumerate(prq):
        prec = prq[i, :, 0]
        rec = prq[i, :, 1]
        quota = prq[i, :, 2]
        if any(quota) < 0.06:
            prq2.append([prec, rec])
        f1[i, :] = 2 * prec * rec / (prec + rec)

    # for score in f1:
    #     print(score)

    pyplot.plot(prq[:, 1, 1], prq[:, 1, 0], color = 'g', label = 'paths', linestyle = 'None',
                marker = '.', markersize = 1.5)
    pyplot.plot(np.mean(prq[:, 1, 1]), np.mean(prq[:, 1, 0]), color = 'magenta', label = 'mean paths',
                linestyle = 'None',
                marker = '.', markersize = 5)
    pyplot.legend()
    pyplot.savefig(base_dir + 'por_path' + tested_name + plotpng)
    tikz.save(base_dir + 'por_path' + tested_name + plottex)
    pyplot.show()
    pyplot.plot(prq[:, 0, 1], prq[:, 0, 0], color = 'r', label = 'objects', linestyle = 'None',
                marker = '.', markersize = 1.5)
    pyplot.plot(np.mean(prq[:, 0, 1]), np.mean(prq[:, 0, 0]), color = 'cyan', label = 'mean objects',
                linestyle = 'None',
                marker = '.', markersize = 5)
    pyplot.legend()
    pyplot.savefig(base_dir + 'por_obj' + tested_name + plotpng)
    tikz.save(base_dir + 'por_obj' + tested_name + plottex)
    pyplot.show()
    pyplot.plot(prq[:, 2, 1], prq[:, 2, 0], color = 'b', label = 'unknown', linestyle = 'None',
                marker = '.', markersize = 1.5)
    pyplot.plot(np.mean(prq[:, 2, 1]), np.mean(prq[:, 2, 0]), color = 'magenta', label = 'mean unknown',
                linestyle = 'None',
                marker = '.', markersize = 5)
    pyplot.savefig(base_dir + 'por_unknown' + tested_name + plotpng)
    tikz.save(base_dir + 'por_unknown' + tested_name + plottex)
    pyplot.legend()
    pyplot.show()
    mean_prec_green = np.mean(prq[:, 1, 1])
    mean_rec_green = np.mean(prq[:, 1, 0])
    mean_prec_red = np.mean(prq[:, 0, 1])
    mean_rec_red = np.mean(prq[:, 0, 0])
    mean_prec_blue = np.mean(prq[:, 2, 1])
    mean_rec_blue = np.mean(prq[:, 2, 0])
    print('mean paths prec: %.3f | rec %.3f' % (mean_prec_green, mean_rec_green))
    print('mean obj prec: %.3f | rec %.3f' % (np.mean(prq[:, 0, 1]), np.mean(prq[:, 0, 0])))
    print('mean unknown prec: %.3f | rec %.3f' % (np.mean(prq[:, 2, 1]), np.mean(prq[:, 2, 0])))
    f1_green = 2 * mean_prec_green *mean_rec_green/ (mean_rec_green+mean_prec_green)
    f1_red = 2 * mean_prec_red *mean_rec_red/ (mean_rec_red+mean_prec_red)
    f1_blue = 2 * mean_prec_blue *mean_rec_blue/ (mean_rec_blue+mean_prec_blue)

    f1_scores = np.array([f1_green, f1_red, f1_blue])
    np.savetxt(base_dir + 'f1_' + tested_name + '.txt', f1_scores)



    n, bins, patches = pyplot.hist(f1[:, 0], 50, facecolor = 'r')
    pyplot.xlabel('F1-Score')
    pyplot.ylabel('Occurances')
    pyplot.title('Histogram of F1Score for object detection')
    pyplot.grid(True)
    pyplot.savefig(base_dir + 'hist_obj' + tested_name + plotpng)
    tikz.save(base_dir + 'hist_obj' + tested_name + plottex)
    pyplot.show()
    n, bins, patches = pyplot.hist(f1[:, 2], 50, facecolor = 'b')
    pyplot.xlabel('F1-Score')
    pyplot.ylabel('Occurances')
    pyplot.title('Histogram of F1Score for unknown area')
    pyplot.grid(True)
    pyplot.savefig(base_dir + 'hist_unknown' + tested_name + plotpng)
    tikz.save(base_dir + 'hist_unknown' + tested_name + plottex)
    pyplot.show()
    n, bins, patches = pyplot.hist(f1[:, 1], 50, facecolor = 'g')

    pyplot.xlabel('F1-Score')
    pyplot.ylabel('Occurances')
    pyplot.title('Histogram of F1Score for paths')
    pyplot.grid(True)
    pyplot.savefig(base_dir + 'hist_path' + tested_name + plotpng)
    tikz.save(base_dir + 'hist_path' + tested_name + plottex)
    pyplot.show()


if __name__ == "__main__":
    print('started')
    t = time.time()
    try:
        main()
        e = time.time()
        print('elapsed time %.3f s' % (e - t))
        print('evaluated without errors')
    except KeyboardInterrupt:
        print("Finished by user")
        e = time.time()
        print('elapsed time %.3f s' % (e - t))
