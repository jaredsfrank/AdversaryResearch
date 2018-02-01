"""Runs adversarial example trainer

Example Usage:
python fgsm_main.py --target_class 0 \
--image_reg 100 --lr .0243 --cuda --batch_size 20


"""
import argparse
import FGSM
import numpy as np
import queue

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", 
    help="Number of adversarial examples to generate",
    type=int)
parser.add_argument("--target_class",
    help="Target class",
    type=int)
parser.add_argument("--image_reg",
    help="Regularizer on the image loss function",
    type=int)
parser.add_argument("--lr",
    help="Learning rate", 
    type=float)
parser.add_argument('--verbose', action='store_true', help='print messages?')
parser.add_argument('--show_images', action='store_true', help='show images?')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
args = parser.parse_args()

def better_range(min_value, max_value):
    l = [min_value, max_value]
    q = Queue.Queue()
    q.put((min_value+1, max_value))
    seen = set()
    while not q.empty():
        next_min, next_max = q.get()
        next_mid = (next_min + next_max)/2
        l.append(next_mid)
        if next_max - next_min > 1:
            if next_mid - next_min > 0:
                q.put((next_min, next_mid))
            if next_max - next_mid > 1:
                q.put((next_mid+1, next_max))
    return l

if __name__ == "__main__":
    fgsm = FGSM.FGSM(args.batch_size)
    ave_mse = 0.0
    iters = 10
    if args.verbose:
        fgsm.verbose = True
    if args.show_images:
        fgsm.show_images = True
    if args.cuda:
        fgsm.cuda = True
    print(fgsm.create_one_adversary_batch(args.target_class, args.image_reg, args.lr))

    # if not args.lr:
    #     save_result = []
    #     np.savetxt("/scratch/jsf239/fgsm_results.csv", np.array(save_result))
    #     for i in better_range(1, 9):
    #         ave_mse, succ = fgsm.create_all_adversaries(target_class=args.target_class,
    #                                                    image_reg=args.image_reg,
    #                                                    lr=(3**i * 10**-4))
    #         save_result.append([ave_mse, succ])
    #         np.savetxt("/scratch/jsf239/fgsm_results.csv", np.array(save_result))
    #     print(ave_mse)
    # else:
    #     ave_mse, succ = fgsm.create_all_adversaries(target_class=args.target_class,
    #                                                    image_reg=args.image_reg,
    #                                                    lr=args.lr)
    #     print(ave_mse, succ)

    
