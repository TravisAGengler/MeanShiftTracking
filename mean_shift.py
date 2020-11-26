#!/usr/bin/env python3

import argparse
import cv2
import io
import json
import numpy as np
import os
import re
import zipfile

from datetime import datetime

# Constants
BIT_DEPTH = 8
FRAMERATE = 15
N_BINS = 32

def write_results(results, results_dir):
  print(f"Writing results to: {results_dir}")
  timestamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
  result_out_dir = os.path.join(results_dir, timestamp + '-' + results['name'])
  if not os.path.exists(result_out_dir):
    print(f'Creating output directory {result_out_dir}')
    # TODO: Write results
    #os.mkdir(result_out_dir)

def get_frame_count(seq):
  info = seq.namelist()
  # Get the number of names that have .jpg extensions
  r = re.compile(".*\.jpg")
  list_pre = list(filter(r.match, info))
  # TRICKY: Some of the datasets contain a __MACOSX folder that has metadata with same filename...
  list_post = [frame for frame in list_pre if '__' not in frame]
  n_frames = len(list_post)
  return n_frames

def read_frame(seq, frame_num):
  info = seq.namelist()
  frame_str = str(frame_num+1).zfill(4)
  search_str = f".*{frame_str}\.jpg"
  r = re.compile(search_str)
  frame_name = list(filter(r.match, info))
  if len(frame_name) < 1:
    print(f"Couldn't find frame {frame_str}")
    exit()
  img_data = seq.read(frame_name[0])
  img_bytes = np.frombuffer(img_data, np.uint8)
  img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
  return img

def read_ground_truth(seq):
  info = seq.namelist()
  r = re.compile(".*groundtruth_rect.txt")
  ground_truth_file = list(filter(r.match, info))
  if len(ground_truth_file) < 1:
    print("Couldn't find ground truth file")
    exit()
  gt_data = seq.read(ground_truth_file[0]).decode("utf-8").strip()
  gt_arr = gt_data.split('\n')
  gt = []
  for g in gt_arr:
    bounds = g.split(',')
    if len(bounds) != 4:
      bounds = g.split('\t')
    gt.append({
      'x' : int(bounds[0]),
      'y' : int(bounds[1]),
      'w' : int(bounds[2]),
      'h' : int(bounds[3])
    })
  return gt

# Formula 5
def gauss_kernel_d(x):
  return 0.5*np.exp(-0.5*x)/(2*np.pi)

def gauss_kernel(x):
  return np.exp(-0.5*x)/(2*np.pi)

def get_constrained_bounds(frame, pt, bounds):
  # Make sure that the bounds are constrained to the dimensions of the frame
  h, w, _ = frame.shape

  x_r = pt[0]+(bounds[0]//2)
  x_l = pt[0]-(bounds[0]//2)
  y_d = pt[1]+(bounds[1]//2)
  y_u = pt[1]-(bounds[1]//2)

  c_x = [max(0, x_l), min(w, x_r)]
  c_y = [max(0, y_u), min(h, y_d)]

  return c_x, c_y

def get_binned_vals(vals, space_min, space_max, n_bins):
  binned_vals = []
  for v in vals:
    b_val = int(np.floor(((v - space_min) / space_max) * n_bins))
    binned_vals.append(b_val)
  return binned_vals

def get_pdf_dist(pdf_a, pdf_b):
  return 0

# Formulas 19, 20
def get_pdf(frame, pt, bounds, scale=1):
  scale_bounds = (bounds[0]*scale, bounds[1]*scale)
  pdf = np.zeros((N_BINS, N_BINS, N_BINS))
  C = 0
  bounds_x, bounds_y = get_constrained_bounds(frame, pt, scale_bounds)
  for x in range(bounds_x[0], bounds_x[1]):
    for y in range(bounds_y[0], bounds_y[1]):
        # OpenCV stores images in BGR!
        b, g, r = get_binned_vals(frame[y,x,:], 0, 2**BIT_DEPTH, N_BINS)
        x_dist = (pt[0]-x) / scale_bounds[0]
        y_dist = (pt[1]-y) / scale_bounds[1]
        norm = np.linalg.norm([x_dist, y_dist])**2
        k = gauss_kernel(norm)
        pdf[r, g, b] = pdf[r, g, b] + k
        C = C + k # Accumulate the normalization constant!
  return [p_v / C for p_v in pdf]

def get_target_info(frame, target_rect):
  w = target_rect['w']
  h = target_rect['h']
  # These are centered coordinates!
  x_center = target_rect['x'] - w//2
  y_center = target_rect['y'] - h//2
  return {
    'x_center' : x_center,
    'y_center' : y_center,
    'w' : w,
    'h' : h,
    'pdf' : get_pdf(frame, (x_center, y_center), (w, h))
  }

def mean_shift(frame, target_info):
  # TODO: Pick up here. Look at get_video_and_target_locations2
  return {
      'x' : 0,
      'y' : 0,
      'w' : 20,
      'h' : 20
    }

def my_method(frame, target_info):
  return {
      'x' : 0,
      'y' : 0,
      'w' : 40,
      'h' : 40
    }

def get_frame_results(truth_rect, rects):
  return {}

def draw_rects(img, rects):
  for rect in rects:
    r = rect['rect']
    tl = (r['x'], r['y'])
    br = (r['x'] + r['w'], r['y'] + r['h'])
    color = rect['color']
    thickness = 2
    rect_img = cv2.rectangle(img, tl, br, color, thickness)
  return rect_img

def display_results(frame, rects):
  frame_rects = draw_rects(frame, rects)
  cv2.imshow('',frame_rects)
  cv2.waitKey(1000//FRAMERATE)

def process(sequence_path):
  seq = zipfile.ZipFile(sequence_path, 'r')

  n_frames = get_frame_count(seq)
  ground_truth = read_ground_truth(seq)

  # Get the target_info from the initial frame
  # This corresponds to q in the paper, the target color PDF
  target_info = get_target_info(read_frame(seq, 0), ground_truth[0])

  results = {}
  results['name'] = os.path.basename(sequence_path)

  for frame_num in range(0, n_frames):
    # Get the frame
    frame = read_frame(seq, frame_num)

    # Get the bounding rectangles
    truth_rect = ground_truth[frame_num]
    mean_shift_rect = mean_shift(frame, target_info)
    my_method_rect = my_method(frame, target_info)

    rects = [
      { 'rect': truth_rect, 'color' : (255, 0, 0) },
      { 'rect': mean_shift_rect, 'color' : (0, 255, 0) },
      { 'rect': my_method_rect, 'color' : (0, 0, 255) }
    ]

    # Do the analysis here, accumulate into results
    res = get_frame_results(rects[0], rects[1:])
    results[f'frame_{frame_num}'] = res

    # Show frames here if we want to!
    display_results(frame, rects)

  return results

def main():
  sequence_path, results_dir = parse_args()
  print(f"Doing mean shift on sequence: {sequence_path}")
  results = process(sequence_path)
  write_results(results, results_dir)

def parse_args():
  parser = argparse.ArgumentParser(prog="mean_shift", description='Perform the mean_shift algorithm on the specified sequence')
  parser.add_argument('-s', '--sequence', metavar='PATH', type=str,
                  help='The sequence to perform mean shift on. This is a path to a zip file', required=True)
  parser.add_argument('-o', '--output', metavar='PATH', type=str, nargs='?', default="results",
                  help='The directory to write results to. Subdirectory with format DATE-DATA_SET_NAME will be created')
  args = vars(parser.parse_args())
  return args['sequence'], args['output']

if __name__ == '__main__':
  main()
