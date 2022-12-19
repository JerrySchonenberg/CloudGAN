# python3 -m cloud_removal --img ./SN_PatchGAN/examples/RICE/1/sample.png --target ./SN_PatchGAN/examples/RICE/1/target.png --mask ./SN_PatchGAN/examples/RICE/1/mask.png --output ./SN_PatchGAN/examples/RICE/1/CloudGAN.png --weights_GAN SN_PatchGAN/model_logs/final_RICE --config_GAN SN_PatchGAN/inpaint.yml --weights_AE AE/data_ae/replicate0/checkpoint.h5 --config_AE AE/AE.yml

import os
import cv2
import numpy as np
import neuralgym as ng
import tensorflow as tf

from cloud_detection.networks.autoencoder import AE
from cloud_removal.inpaint_model import InpaintCAModel


def parse_CL():
   import argparse
   parser = argparse.ArgumentParser(add_help=True)

   # SN_PatchGAN ARGUMENTS
   parser.add_argument("--weights_GAN", type=str, required=True,
                       help="Weights of SN-PatchGAN model")
   parser.add_argument("--config_GAN", type=str, required=True,
                       help="Config file (.yml) of SN-PatchGAN")
   # AUTO-ENCODER ARGUMENTS
   parser.add_argument("--weights_AE", type=str, required=True,
                       help="Weights of AE model")
   parser.add_argument("--config_AE", type=str, required=True,
                       help="Config file (.yml) of AE")
   # FILENAME ARGUMENTS
   parser.add_argument("--img", type=str, required=True,
                       help="Location of image to be processed")
   parser.add_argument("--no_AE", action="store_true",
                       help="Do not use AE but use given mask (via argument '--mask')")
   parser.add_argument("--target", type=str, default=None,
                       help="Ground truth for input image")
   parser.add_argument("--output", type=str, default="out.png",
                       help="Filename of output")
   parser.add_argument("--mask", type=str, default=None,
                       help="Filename of mask (default does not save mask)")
   # MISC. ARGUMENTS
   parser.add_argument("--debug", action="store_true",
                       help="Enable debug-messages from TensorFlow")
   
   return parser.parse_args()


def load_configs(config_AE, config_GAN):
   import yaml
   assert os.path.exists(config_AE) and os.path.exists(config_GAN)
   with open(config_AE, 'r') as c_AE, open(config_GAN, 'r') as c_GAN:
      FLAGS_AE = yaml.load(c_AE, Loader)
      FLAGS_GAN = yaml.load(c_GAN, Loader)
   return FLAGS_AE, FLAGS_GAN


def get_AE(weights, FLAGS):
   model = AE(FLAGS.input_res, FLAGS.activation, FLAGS.activation_out)
   model.build(input_shape=(1, FLAGS.input_res, FLAGS.input_res, 3))
   model.load_weights(weights)
   return model


def remove_clouds(img, args):
   # Load hyperparameter configs for AE and SN_PatchGAN
   FLAGS_AE, FLAGS_GAN = ng.Config(args.config_AE), ng.Config(args.config_GAN)

   # Generate cloud mask   
   if not args.no_AE:
      print("Detecting clouds...")
      img_AE = np.expand_dims(img * 1./255, axis=0)
      with tf.compat.v1.Session() as sess:
         img_AE = tf.constant(img_AE, dtype=tf.float32)  # Normalize image
         AE = get_AE(args.weights_AE, FLAGS_AE)
         mask = AE.to_hardmask(sess.run(AE(img_AE, training=False)))[0]
         
         if args.mask is not None:  # Save hardmask
            cv2.imwrite(args.mask, np.multiply(mask, 255))  # Convert back to [0,255] range
         mask = np.repeat(mask[:, :], 3, axis=2)  # Convert to 3-channels
   else:  # Use precomputed mask, so no AE used
      print("Using given cloud mask...")
      assert os.path.exists(args.mask)
      mask = cv2.imread(args.mask) * 1./255
      assert mask is not None

   # Prepare image and mask for SN_PatchGAN
   img = np.expand_dims(img, axis=0)
   mask = np.expand_dims(mask, axis=0)
   input_img = np.concatenate([img, mask * 255], axis=2)

   # Prepare session and graph
   tf.compat.v1.reset_default_graph()
   sess_config = tf.compat.v1.ConfigProto()
   sess_config.gpu_options.allow_growth = True

   # Cloud inpainting
   print("Removing clouds...")
   SN_PatchGAN = InpaintCAModel()
   with tf.compat.v1.Session(config=sess_config) as sess:
      # Prepare TF graph
      input_img = tf.constant(input_img, dtype=tf.float32)
      output = SN_PatchGAN.build_server_graph(FLAGS_GAN, input_img)
      output = (output + 1.) * 127.5
      output = tf.reverse(output, [-1])
      output = tf.saturate_cast(output, tf.uint8)

      # Load checkpoint SN_PatchGAN
      vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
      assign_ops = []
      for var in vars_list:
         vname = var.name
         var_value = tf.contrib.framework.load_variable(args.weights_GAN, vname)
         assign_ops.append(tf.assign(var, var_value))
      
      # Process image + mask and save output
      sess.run(assign_ops)
      output_img = sess.run(output)[0][:, :, ::-1]
      cv2.imwrite(args.output, output_img)

      if args.target:  # Compute SSIM and PSNR given the target image and prediction
         assert os.path.exists(args.target)
         from skimage.metrics import structural_similarity, peak_signal_noise_ratio
         target = cv2.imread(args.target)
         SSIM = structural_similarity(target, output_img, multichannel=True, K1=0.01, K2=0.03)
         PSNR = peak_signal_noise_ratio(target, output_img)
         print(f"Quality metrics ->  SSIM: {round(SSIM, 4)} | PSNR: {round(PSNR, 4)} dB")
      
      print("Process completed!")


if __name__ == "__main__":
   args = parse_CL()

   if not args.debug:
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
      tf.get_logger().setLevel('ERROR')

   print("\n\n------------- CLOUD REMOVAL -------------")
   print("Loading image...")

   assert os.path.exists(args.img)
   img = cv2.imread(args.img)
   assert img is not None  # Loading image with opencv did not succeed

   remove_clouds(img, args)
