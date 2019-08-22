import numpy as np
import multiprocessing
import time
from matplotlib import pyplot
from evaluator import Evaluator
from diagnostics import calc_best_th

prec_over_rec = np.zeros((256, 3, 2), dtype = np.float)

base_dir = '/media/localadmin/BigBerta/11Nils/kitti/dataset/sequences/08/'
# inf_dir = '20190808_160355/'
# inf_dir = '20190730_170755/'
inf_dir = 'inference/'
label_dir = 'labels_gt/'
img_dir = 'image_2/'

num_img = 100
lower_bound = 0


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main():

    evaluator = Evaluator(base_dir, img_dir, inf_dir, label_dir, lower_bound, num_img)

    num_proc = 20
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
    try:
        print('started')
        t = time.time()
        main()
        e = time.time()
        print('elapsed time %.3f s' % (e - t))
    except KeyboardInterrupt:
        print("Finished")
