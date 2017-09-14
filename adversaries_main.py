"""Runs adversarial example trainer

Example Usage:
python adversaries_main.py --batch_size 100 \
    --target_class 1 --image_reg 100 --lr 1


"""
import argparse
import model_testing
import adversaries

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
args = parser.parse_args()


if __name__ == "__main__":
    lbfgs = adversaries.LBFGS()
    ave_mse = 0.0
    iters = 10
    if args.verbose:
        lbfgs.verbose = True
    for i in range(iters):
        mse = lbfgs.create_adversary(batch_size=args.batch_size,
                                     target_class=args.target_class,
                                     image_reg=args.image_reg,
                                     lr=args.lr)
        
        ave_mse += mse.data.cpu().numpy()[0]
        print "At iteration {}, the average mse is {}".format(i, ave_mse/float(iters))
    print ave_mse/float(iters)
    
