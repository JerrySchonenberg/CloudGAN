import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Dataset:
   def __init__(self, args):
      self.datagen_args = dict(
         rescale=1./255,
         rotation_range=180,
         horizontal_flip=True,
         vertical_flip=True,
         width_shift_range=0.1,
         height_shift_range=0.1,
         fill_mode="reflect",
         zoom_range=0.2
      )

      def create_generator(subset, args, augmentation=False):
         print(f"\nLoad data from {subset}:")

         datagen = self.datagen_args if augmentation else dict(rescale=1./255)
         
         sample_datagen = ImageDataGenerator(**datagen)
         mask_datagen = ImageDataGenerator(**datagen)

         sample_generator = sample_datagen.flow_from_directory(
            subset+"img",
            target_size=(args.input_res, args.input_res),
            color_mode="rgb",
            class_mode=None,
            shuffle=True,
            seed=args.seed,
            batch_size=args.batch_size
         )
         mask_generator = sample_datagen.flow_from_directory(
            subset+"mask",
            target_size=(args.input_res, args.input_res),
            color_mode="grayscale",
            class_mode=None,
            shuffle=True,
            seed=args.seed,
            batch_size=args.batch_size
         )
         return (pair for pair in zip(sample_generator, mask_generator)), sample_generator.__len__()

      self.train_generator, self.n_batches_train = create_generator(args.dataset_train, args, args.augmentation)
      self.test_generator,  self.n_batches_test  = create_generator(args.dataset_test,  args)
   
   def get_datagen_args(self):
      return self.datagen_args