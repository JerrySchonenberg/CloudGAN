# Parses the command line arguments
def parse_CL():
   import argparse
   parser = argparse.ArgumentParser(add_help=True)

   # General arguments
   parser.add_argument("--seed", type=int, default=42, 
                       help="seed to initialize training")
   parser.add_argument("--weights", type=str, default=None, 
                       help="location of pretrained weights to be used as initialization")
   parser.add_argument("--demo", action="store_true", 
                       help="perform a demo run")
   parser.add_argument("--summary", action="store_true",
                       help="show summary of AE architecture")
   parser.add_argument("--analyse", type=str, default=None,
                       help="analyse replications in given folder")
   
   parser.add_argument("--model", type=str, default="AE", choices=["AE", "UNET"],
                       help="select model architecture (AE, UNET)")

   # Logger arguments
   parser.add_argument("--save_dir", type=str, default=None, 
                       help="directory where output data is stored")
   
   # Dataset arguments
   parser.add_argument("--dataset_train", type=str, default="../../datasets/38-cloud/train/",
                       help="dataset to train AE on")
   parser.add_argument("--dataset_test", type=str, default="../../datasets/38-cloud/test/",
                       help="dataset to test AE on")
   parser.add_argument("--augmentation", action="store_true",
                       help="enabling data augmentation during training")

   # Training arguments
   parser.add_argument("--epochs", type=int, default=64,
                       help="number of epochs the model is trained")
   parser.add_argument("--batch_size", type=int, default=32,
                       help="number of samples in one batch")
   parser.add_argument("--workers", type=int, default=1,
                       help="number of workers during training")
   parser.add_argument("--mp", action="store_true",
                       help="use multiprocessing during training")
   
   # Autoencoder architecture arguments
   parser.add_argument("--input_res", type=int, default=256,
                       help="resolution of input images (res x res)")
   parser.add_argument("--activation", type=str, default="relu", 
                       help="activation function of conv layers")
   parser.add_argument("--activation_out", type=str, default="sigmoid",
                       help="activation function of output layer")
   parser.add_argument("--optimizer", type=str, default="Adam",
                       help="optimizer of AE")
   parser.add_argument("--lr", type=float, default=1e-3,
                       help="learning rate of optimizer")
   parser.add_argument("--loss", type=str, default="MeanSquaredError",
                       help="loss function used by optimizer of AE")
   
   return parser.parse_args()

if __name__ == "__main__":
   args = parse_CL()
   
   from trainer import Trainer
   T = Trainer(args)
   
   if args.demo:
      T.demo(args)
   elif args.analyse is not None:
      T.analyse(args)
   else:
      T.train()

   T.close_session()
