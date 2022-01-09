import matplotlib.pyplot as plt
import cv2

N = 4  # Number of examples

if __name__ == "__main__":
   fig, ax = plt.subplots(N,7)

   for i in range(N):
      directory = str(i+1)

      img = cv2.imread(f"./{directory}/sample.png")
      target = cv2.imread(f"./{directory}/target.png")
      mask = cv2.imread(f"./{directory}/mask.png")
      mask_img = cv2.imread(f"./{directory}/mask_img.png")
      target_img = cv2.imread(f"./{directory}/mask_target.png")
      GAN_yes = cv2.imread(f"./{directory}/CloudGAN-yes.png")
      GAN_no = cv2.imread(f"./{directory}/CloudGAN-no.png")
      
      ax[i,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      ax[i,0].set_xticks([])
      ax[i,0].set_yticks([])

      ax[i,1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
      ax[i,1].set_xticks([])
      ax[i,1].set_yticks([])

      ax[i,2].imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))
      ax[i,2].set_xticks([])
      ax[i,2].set_yticks([])

      ax[i,3].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
      ax[i,3].set_xticks([])
      ax[i,3].set_yticks([])

      ax[i,4].imshow(cv2.cvtColor(GAN_yes, cv2.COLOR_BGR2RGB))
      ax[i,4].set_xticks([])
      ax[i,4].set_yticks([])

      ax[i,5].imshow(cv2.cvtColor(GAN_no, cv2.COLOR_BGR2RGB))
      ax[i,5].set_xticks([])
      ax[i,5].set_yticks([])

      ax[i,6].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
      ax[i,6].set_xticks([])
      ax[i,6].set_yticks([])
   
   ax[0,0].set_title("Image")
   ax[0,1].set_title("Hard mask - AE")
   ax[0,2].set_title(r"Input + mask ($\bf{1}$)")
   ax[0,3].set_title("Ground Truth\n"+r"+ mask ($\bf{2}$)")
   ax[0,4].set_title(r"CloudGAN ($\bf{1}$)")
   ax[0,5].set_title(r"CloudGAN ($\bf{2}$)")
   ax[0,6].set_title("Ground Truth")

   plt.show()