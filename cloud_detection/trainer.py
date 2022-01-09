import tensorflow as tf
from tensorflow import keras

from logger import Logger
from autoencoder import AE
from unet import Unet
from dataset import Dataset
from plotter import Plotter
from metrics import BinaryTP, BinaryFP, BinaryTN, BinaryFN


class Trainer:
   def __init__(self, args):
      self.epochs = args.epochs
      self.batch_size = args.batch_size
      self.workers = args.workers
      self.mp = args.mp

      self.sess = tf.compat.v1.Session()
      tf.compat.v1.keras.backend.set_session(self.sess)
      tf.compat.v1.set_random_seed(args.seed)

      if args.model == "AE":
         self.model = AE(
            args.input_res, 
            args.activation, 
            args.activation_out, 
            args.seed
         )
      elif args.model == "UNET":
         self.model = Unet(
            args.input_res, 
            args.activation, 
            args.activation_out, 
            args.seed
         )
      else:
         print(f"ERROR: {args.model} is an invalid option.")
         self.close_session()
         exit()

      self.model.build((None, args.input_res, args.input_res, 3))
      
      self.logger = Logger(args)
      self.dataset = Dataset(args)

      if not args.weights == None:
         self.model.load_weights(args.weights)
      
      self.optimizer = tf.keras.optimizers.get({"class_name": args.optimizer,
                                                "config": {"learning_rate": args.lr}})
      self.loss_fn = tf.keras.losses.get(args.loss)
      
      self.model.compile(loss=self.loss_fn,
                         optimizer=self.optimizer,
                         metrics=[BinaryTP(), 
                                  BinaryFP(), 
                                  BinaryTN(), 
                                  BinaryFN()])
      if args.summary: self.model.summary()

   def close_session(self):
      self.sess.close()

   def train(self):
      model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
         filepath=self.logger.get_checkpoint_path(),
         save_weights_only=True,
         save_best_only=False,
         save_freq="epoch"
      )

      history = self.model.fit_generator(
         self.dataset.train_generator,
         validation_data=self.dataset.test_generator,
         validation_steps=self.dataset.n_batches_test,
         steps_per_epoch=self.dataset.n_batches_train,
         epochs=self.epochs,
         workers=self.workers,
         use_multiprocessing=self.mp,
         callbacks=[model_checkpoint_callback]
      )
      
      evaluate = self.model.evaluate(
         self.dataset.test_generator,
         steps=self.dataset.n_batches_test
      )

      self.logger.save_hist(history.history)
      self.logger.save_eval(evaluate)

   # Generate plots showing the various aspects of one model
   def demo(self, args):
      P = Plotter(self.model, self.dataset, self.sess)
      
      P.plot_grid(5, self.dataset.test_generator)
      
      assert not args.augmentation  # Otherwise img is already augmented for Plotter
      (sample_batch, mask_batch) = next(self.dataset.train_generator)
      P.plot_augmentation(5, sample_batch[0], mask_batch[0])
   
   # Generate plots showing training trajectory over multiple replications
   def analyse(self, args):
      P = Plotter(self.model, self.dataset, self.sess)
      
      P.plot_training_process(args.analyse, self.logger.FILE_LOG_HIST, "loss")