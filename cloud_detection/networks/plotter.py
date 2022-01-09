import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as tf_image


class Plotter:
   FILE_GRID = "grid.svg"
   FILE_AUG_IMG = "aug_img.svg"
   FILE_AUG_MASK = "aug_mask.svg"

   def __init__(self, AE, dataset, sess):
      self.AE = AE
      self.dataset = dataset
      self.sess = sess
   
   # Plot Nx4 grid of examples from datagen 
   def plot_grid(self, N, datagen):
      (sample_batch, mask_batch) = next(datagen)
      fig, ax = plt.subplots(N, 4)

      assert N <= len(sample_batch)

      # Combine prediction hard mask and ground truth
      def combine_img(pred, target):
         A = np.full((256, 256, 3), 0)   # Black output

         C1 = [255, 255, 255]   # 1 in target, 1 in pred
         C2 = [0, 192, 222]     # 1 in target, 0 in pred
         C3 = [255, 0, 0]       # 0 in target, 1 in pred
         
         for y in range(len(pred)):
            for x in range(len(pred)):
               if pred[y][x] == target[y][x] and pred[y][x] == 1:
                  A[y][x] = C1
               elif pred[y][x] != target[y][x] and pred[y][x] == 0:
                  A[y][x] = C2
               elif pred[y][x] != target[y][x] and pred[y][x] == 1:
                  A[y][x] = C3
         return A

      pred = self.sess.run(self.AE(sample_batch[:N], training=False))
      for i in range(N):
         img = sample_batch[i]
         target = mask_batch[i]

         # Input image
         ax[i,0].imshow(img)
         ax[i,0].set_xticks([])
         ax[i,0].set_yticks([])

         # Target mask
         ax[i,1].imshow(target, cmap="gray")
         ax[i,1].set_xticks([])
         ax[i,1].set_yticks([])

         # Predicted soft mask
         ax[i,2].imshow(pred[i], cmap="gray")
         ax[i,2].set_xticks([])
         ax[i,2].set_yticks([])

         # Prediction converted into hard mask
         hardmask = self.AE.to_hardmask(pred[i])
         combine = combine_img(hardmask, target)
         ax[i,3].imshow(combine)
         ax[i,3].set_xticks([])
         ax[i,3].set_yticks([])

      ax[0,0].set_title("Image")
      ax[0,1].set_title("Ground Truth")
      ax[0,2].set_title("Soft mask\nPrediction")
      ax[0,3].set_title("Hard mask\nPrediction")
      plt.show()
      #plt.savefig(self.FILE_GRID, bbox_inches="tight")

   # Plot NxN grid of different augmentations of the same image
   def plot_augmentation(self, N, img, mask):
      import random

      def generate_plot(N, tensor, seeds, filename, datagen, cmap=None):
         fig, ax = plt.subplots(N, N)
         for i in range(N):
            for j in range(N):
               aug = datagen.random_transform(tensor, seed=seeds[i*N + j])

               ax[i,j].set_xticks([])
               ax[i,j].set_yticks([])
               ax[i,j].imshow(aug, cmap=cmap)
         
         plt.savefig(filename, bbox_inches="tight")

      seeds = random.sample(range(1000), N**2)

      datagen = tf_image.ImageDataGenerator(**self.dataset.get_datagen_args())
      generate_plot(N, img, seeds, self.FILE_AUG_IMG, datagen)
      generate_plot(N, mask, seeds, self.FILE_AUG_MASK, datagen, "gray")

   # Plot trajectory of loss on training and test set over N replications
   # Possible metrics: loss, precision, recall, F1-score
   # To get test metrics, add "val_" as prefix to metric
   def plot_training_process(self, data_dir, filename, metric):
      import os
      import numpy as np

      # Do some optional preprocessing of datapoints (in case of recall, precision, F1)
      def get_datapoint(rep, i, metric, test=False):
         if metric == "loss":
            return data[rep]["val_"+metric][i] if test else data[rep][metric][i]
         else:  # Recall, Precision or F1-score
            TP = data[rep]["val_binary_TP"][i] if test else data[rep]["binary_TP"][i]
            FP = data[rep]["val_binary_FP"][i] if test else data[rep]["binary_FP"][i]
            FN = data[rep]["val_binary_FN"][i] if test else data[rep]["binary_FN"][i]

            if metric == "precision":
               return TP / (TP + FP)
            elif metric == "recall":
               return TP / (TP + FN)
            elif metric == "F1-score":
               return TP / (TP + 0.5 * (FP + FN))
      
      # Plot line with confidence interval
      def plot(X, data, label, color):
         Y1 = [i[0] for i in data]  # AVG
         CI = [i[1] for i in data]  # CI

         Y2 = [x + y for x, y in zip(Y1, CI)]  # Upper CI
         Y3 = [x - y for x, y in zip(Y1, CI)]  # Lower CI

         plt.plot(X, Y1, label=label, color=color)
         plt.fill_between(X, y1=Y2, y2=Y3, color=color, alpha=0.3)


      rep_dirs = [next(os.walk(data_dir))[1]][0]
      N_REP = len(rep_dirs)

      # Load history data
      data = [np.load(os.path.join(data_dir, rep_dirs[i], filename), allow_pickle=True).item() \
              for i in range(N_REP)]
      
      X = range(len(data[0]["loss"]))
      Y1, Y2 = [], []

      # Compute AVG and CI for training and test sets
      for i in X:
         SUM_TRAIN, SUM_TEST = 0., 0.

         for rep in range(N_REP):
            SUM_TRAIN += get_datapoint(rep, i, metric)
            SUM_TEST  += get_datapoint(rep, i, metric, True)
         
         AVG_TRAIN = round(SUM_TRAIN / N_REP, 4)
         AVG_TEST  = round(SUM_TEST / N_REP, 4)

         DIST_TRAIN, DIST_TEST = 0., 0.
         for rep in range(N_REP):
            DIST_TRAIN += (get_datapoint(rep, i, metric) - AVG_TRAIN)**2
            DIST_TEST += (get_datapoint(rep, i, metric, True) - AVG_TEST)**2
         
         CI_TRAIN = np.sqrt(DIST_TRAIN / N_REP) / np.sqrt(N_REP)
         CI_TEST = np.sqrt(DIST_TEST / N_REP) / np.sqrt(N_REP)

         Y1.append((AVG_TRAIN, CI_TRAIN))
         Y2.append((AVG_TEST, CI_TEST))

      # Plot the training and test performance
      plot(X, Y1, "Train set", "navy")
      plot(X, Y2, "Test set", "darkred")

      plt.xlabel("Epoch")
      plt.ylabel("MSE loss")#metric.capitalize())
      plt.legend()
      plt.grid(b=True)
      plt.xlim(0, max(X))
      plt.title(f"MSE loss throughout the training process (3 replications)")
      plt.savefig(os.path.join(data_dir, metric+".svg"), bbox_inches="tight")