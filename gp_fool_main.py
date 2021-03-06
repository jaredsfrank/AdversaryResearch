"""Runs adversarial example trainer

Example Usage:
python fgsm_main.py --batch_size 100 \
    --target_class 1 --image_reg 100


"""
import argparse
import gp_fool
import numpy as np

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


if __name__ == "__main__":
    gpfool = gp_fool.GPFool(args.batch_size)
    ave_mse = 0.0
    iters = 10
    if args.verbose:
        gpfool.verbose = True
    if args.show_images:
        gpfool.show_images = True
    if args.cuda:
        gpfool.cuda = True

    ave_mse, succ = gpfool.create_all_adversaries(target_class=args.target_class,
                                               image_reg=args.image_reg,
                                               lr=args.lr)
    print(ave_mse)
    
