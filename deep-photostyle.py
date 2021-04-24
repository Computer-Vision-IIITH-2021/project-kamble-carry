import argparse
from PIL import Image
import numpy as np
import os
from photo_style import stylize

parser = argparse.ArgumentParser()
# Input Options
parser.add_argument("--content_image_path", dest='content_image_path',  nargs='?',
                    help="Path to the content image")
parser.add_argument("--style_image_path",   dest='style_image_path',    nargs='?',
                    help="Path to the style image")
parser.add_argument("--content_seg_path",   dest='content_seg_path',    nargs='?',
                    help="Path to the style segmentation")
parser.add_argument("--style_seg_path",     dest='style_seg_path',      nargs='?',
                    help="Path to the style segmentation")
parser.add_argument("--init_image_path",    dest='init_image_path',     nargs='?',
                    help="Path to init image", default="")
parser.add_argument("--output_image",       dest='output_image',        nargs='?',
                    help='Path to output the stylized image', default="best_stylized.png")
parser.add_argument("--serial",             dest='serial',              nargs='?',
                    help='Path to save the serial out_iter_X.png', default='./')

# Training Optimizer Options
parser.add_argument("--max_iter",           dest='max_iter',            nargs='?', type=int,
                    help='maximum image iteration', default=1000)
parser.add_argument("--learning_rate",      dest='learning_rate',       nargs='?', type=float,
                    help='learning rate for adam optimizer', default=1.0)
parser.add_argument("--print_iter",         dest='print_iter',          nargs='?', type=int,
                    help='print loss per iterations', default=1)