"""Runs adversarial example trainer

Example Usage:
python adversaries_main.py 4 1 100 0.01


"""
import argparse
import model_testing
import adversaries

parser = argparse.ArgumentParser()
parser.add_argument("batch_size", 
    help="Number of adversarial examples to generate",
    type=int)
parser.add_argument("target_class",
    help="Target class",
    type=int)
parser.add_argument("image_reg",
    help="Regularizer on the image loss function",
    type=int)
parser.add_argument("lr",
    help="Learning rate", 
    type=float)
args = parser.parse_args()


if __name__ == "__main__":
    # model_testing.create_adversary(batch_size=args.batch_size,
    #                                target_class=args.target_class,
    #                                image_reg=args.image_reg,
    #                                lr=args.lr,
    #                                l_inf=False)
    lbfgs = adversaries.LBFGS()
    ave_mse = 0.0
    iters = 1
    for i in range(iters):
        mse = lbfgs.create_adversary(batch_size=args.batch_size,
                                     target_class=args.target_class,
                                     image_reg=args.image_reg,
                                     lr=args.lr)
        print mse
        print mse.numpy()
        # ave_mse += mse
    print 
    
