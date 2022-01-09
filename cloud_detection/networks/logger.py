import os
import json
import numpy as np
from datetime import datetime


class Logger:
   FILE_LOG_HIST = "history.npy"
   FILE_LOG_EVAL = "eval.csv"
   FILE_WEIGHTS  = "checkpoint.h5"
   FILE_ARGS     = "config.json"

   def __init__(self, args):
      if not args.demo and args.analyse is None:
         if args.save_dir == None:
            self.dir = datetime.now().strftime("data_%d-%m-%Y_%H:%M:%S")
         else:
            self.dir = args.save_dir
         os.mkdir(self.dir)
         self.save_args(args)
   
   def get_checkpoint_path(self):
      return os.path.join(self.dir, self.FILE_WEIGHTS)
   
   def save_args(self, args):
      with open(os.path.join(self.dir, self.FILE_ARGS), 'w') as f:
         json.dump(args.__dict__, f, indent=3)

   def save_hist(self, history):
      np.save(os.path.join(self.dir, self.FILE_LOG_HIST), history)

   def save_eval(self, evaluate):
      with open(os.path.join(self.dir, self.FILE_LOG_EVAL), 'w') as f:
         f.write("loss,binary_TP,binary_FP,binary_TN,binary_FN,precision,recall,F1-score\n")
         for i in range(len(evaluate)):
            f.write(str(evaluate[i])+',')
         
         TP, FP, FN = evaluate[1], evaluate[2], evaluate[4]
         f.write(str(TP / (TP + FP))+',')           # Precision
         f.write(str(TP / (TP + FN))+',')           # Recall
         f.write(str(TP / (TP + 0.5 * (FP + FN))))  # F1-score