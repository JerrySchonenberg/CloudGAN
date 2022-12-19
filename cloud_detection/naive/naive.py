import cv2
import os
import numpy as np

from tqdm import tqdm


RES = 256  # Resolution of images

# Location of 38-Cloud data
TRAIN = "../../src/datasets/38-cloud/train/"
TEST  = "../../src/datasets/38-cloud/test/"
IMG   = "img/data/"
MASK  = "mask/data/"


# Plot the probabilities
def plot(H, S, V):
   import matplotlib.pyplot as plt

   X = range(256)
   # Plot smoothened probability graphs for H, S and V
   plt.plot(X, H, color="tab:blue", label="Hue", zorder=3)
   plt.plot(X, S, color="tab:red", label="Saturation", zorder=3, linestyle=(0, (1,1)))
   plt.plot(X, V, color="tab:green", label="Value", zorder=3, linestyle="dashed")

   plt.ylabel("Probability")
   plt.xlabel("Value of component")
   plt.title("Cloud probability per HSV-component over the train set")

   plt.legend()
   plt.xlim(0, 255)
   plt.ylim(0, 1)
   plt.grid(zorder=0)
   plt.show()


# Process train set and compute probabilities
def process_img():
   from scipy.signal import savgol_filter

   print("Computing probabilities over the train set...")

   n_samples = len(os.listdir(TRAIN+IMG))  # Number of samples processed
   H_cloud, S_cloud, V_cloud = np.zeros(256), np.zeros(256), np.zeros(256)  # Count cloud-pixels for every possible value
   H_no, S_no, V_no = np.zeros(256), np.zeros(256), np.zeros(256)  # Count nocloud-pixels for every possible value

   with tqdm(total=n_samples) as bar:
      for img_f, mask_f in zip(os.listdir(TRAIN+IMG), os.listdir(TRAIN+MASK)):
         bar.update(1)

         img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, IMG, img_f)), cv2.COLOR_BGR2HSV)
         mask = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, MASK, mask_f)), cv2.COLOR_BGR2GRAY)

         for y in range(RES):
            for x in range(RES):
               if mask[y][x]:  # Cloud pixel
                  H_cloud[img[y][x][0]] += 1
                  S_cloud[img[y][x][1]] += 1
                  V_cloud[img[y][x][2]] += 1
               else:           # No cloud pixel
                  H_no[img[y][x][0]] += 1
                  S_no[img[y][x][1]] += 1
                  V_no[img[y][x][2]] += 1

   # Compute probabilities and smoothen them
   H = savgol_filter(H_cloud / np.maximum((H_no + H_cloud), np.ones(256)), 11, 4)
   S = savgol_filter(S_cloud / np.maximum((S_no + S_cloud), np.ones(256)), 11, 4)
   V = savgol_filter(V_cloud / np.maximum((V_no + V_cloud), np.ones(256)), 11, 4)

   np.save("H.npy", H, allow_pickle=True)
   np.save("S.npy", S, allow_pickle=True)
   np.save("V.npy", V, allow_pickle=True)

   plot(H, S, V)
   return H, S, V   # Probabilities of being a cloud pixel per HSV-component value


def predict(H, S, V):
   print("Predicting cloud mask on test set...")

   n_samples = len(os.listdir(TEST+IMG))  # Number of samples processed
   TP, FP, TN, FN = 0, 0, 0, 0

   with tqdm(total=n_samples) as bar:
      for img_f, mask_f in zip(os.listdir(TEST+IMG), os.listdir(TEST+MASK)):
         bar.update(1)

         img = cv2.cvtColor(cv2.imread(os.path.join(TEST, IMG, img_f)), cv2.COLOR_BGR2HSV)
         mask = cv2.cvtColor(cv2.imread(os.path.join(TEST, MASK, mask_f)), cv2.COLOR_BGR2GRAY) / 255

         pred = np.zeros((RES, RES))
         for y in range(RES):
            for x in range(RES):
               h = img[y][x][0]
               s = img[y][x][1]
               v = img[y][x][2]

               P = round((H[h] + S[s] + V[v]) / 3)  # Probability to be a cloud, round it to nearest int
               pred[y][x] = P

               if P == 1:  # Prediction -> cloud-pixel
                  if P == mask[y][x]:
                     TP += 1
                  else:
                     FP += 1
               else:  # Prediction -> no-cloud-pixel
                  if P == mask[y][x]:
                     TN += 1
                  else:
                     FN += 1

   precision = TP / (TP + FP)
   recall = TP / (TP + FN)
   F1 = TP / (TP + 0.5 * (FP + FN))
   print("Predictions completed! Results:")
   print(f"Precision: {precision}\nRecall: {recall}\nF1-score: {F1}")


if __name__ == "__main__":
   if not os.path.exists("H.npy"):
      H, S, V = process_img()
   else:
      H = np.load("H.npy", allow_pickle=True)
      S = np.load("S.npy", allow_pickle=True)
      V = np.load("V.npy", allow_pickle=True)

   plot(H, S, V)
   predict(H, S, V)  # Make predictions for testset
