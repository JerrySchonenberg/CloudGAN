import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', default='./img', type=str,
                    help='The folder path to images')
parser.add_argument('--mask_path', default='./mask', type=str,
                    help='The folder path to masks')
parser.add_argument('--out_path', default='./output', type=str,
                    help='The output path to images')
parser.add_argument('--flist_filename', default='./data_flist/batch_test.flist', type=str,
                    help='The batch test flist filename.')

if __name__ == "__main__":

    args = parser.parse_args()

    # get the list of directories
    img = os.listdir(args.img_path) # All images
    mask_dirs = os.listdir(args.mask_path) # List of all bins

    with open(args.flist_filename, 'a+') as fo:
        for image in img: # All images
            for bin in mask_dirs:
                maskbin = args.mask_path + "/" + bin # All masks of current bin
                for mask in os.listdir(maskbin):
                    img_path = args.img_path + "/" + image
                    mask_path = args.mask_path + "/" + bin + "/" + mask
                    output_path = args.out_path + "/" + bin + "/" + image[:-4] + "@" + mask
                    fo.write(img_path + " " + mask_path + " " + output_path + "\n")


    # print process
    print("Written file is: ", args.flist_filename)
