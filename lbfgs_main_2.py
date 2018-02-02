"""Runs adversarial example trainer

Example Usage:
python lbfgs_main_2.py --batch_size 100 \
    --target_class 1 --image_reg 100 --lr 1 --cuda \
    --verbose


"""
import argparse
import LBFGS
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
parser.add_argument("--max_iters",
    help="Maximum number of iterations", 
    type=float)
parser.add_argument('--verbose', action='store_true', help='print messages?')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
args = parser.parse_args()

if __name__ == "__main__":
    lbfgs = LBFGS.LBFGS(args.batch_size)
    lbfgs.show_images = False
    if args.verbose:
        lbfgs.verbose = True
    if args.cuda:
        lbfgs.cuda = True
    if args.max_iters is not None:
        lbfgs.max_iters = args.max_iters
    model = lbfgs.resnet()
    data, adversaries = lbfgs.create_one_adversary_batch(model, target_class=args.target_class, image_reg=args.image_reg, lr=args.lr)
    print(lbfgs.get_stats(data, adversaries, model, targs.target_class))

