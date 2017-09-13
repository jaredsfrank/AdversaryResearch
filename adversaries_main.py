"""Runs adversarial example trainer

Example Usage:
python adversaries_main.py 4 1 100 0.01


"""
import argparse
import model_testing

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
parser.add_argument("--l_inf", 
	help="If flagged, uses l infinity regularizer")
args = parser.parse_args()


if __name__ == "__main__":
	model_testing.create_adversary(batch_size=args.batch_size,
								   target_class=args.target_class,
								   image_reg=args.image_reg,
								   lr=args.lr,
                                                                   l_inf=False)
	
