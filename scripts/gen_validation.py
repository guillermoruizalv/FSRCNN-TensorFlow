import argparse
import numpy as np
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Move a random subset of images from input dir to output dir.')
parser.add_argument('input_dir', help='Input dir containing all the images')
parser.add_argument('output_dir', help='Output dir to generate subset of validation images')
parser.add_argument('p', help='Percentage of images to be moved.')

args = parser.parse_args()

input_files = [f for f in listdir(args.input_dir) if isfile(join(args.input_dir, f))]
print("{} files detected in {}".format(len(input_files), args.input_dir))

output_len = int(len(input_files)*float(args.p))
print("{} files to be moved to {}".format(output_len, args.output_dir))

# Get files
indices = np.arange(len(input_files))
np.random.shuffle(indices)
indices = indices[0:output_len]
output_files = [input_files[i] for i in indices]

# Print linux command
print ("mv {} {}".format(" ".join([args.input_dir+"/"+f for f in output_files]), args.output_dir))
