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
#FRAMERATE = 60
N_BINS = 16
bin_LUT = {}

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

def rect_center(rect):
  return {
    'x_c' : rect['x'] + rect['w']//2,
    'y_c' : rect['y'] + rect['h']//2,
    'w' : rect['w'],
    'h' : rect['h']
  }

def rect_uncenter(rect_c):
  return {
    'x' : rect_c['x_c'] - rect_c['w']//2,
    'y' : rect_c['y_c'] - rect_c['h']//2,
    'w' : rect_c['w'],
    'h' : rect_c['h']
  }

def rect_scale(rect_c, scale):
  return {
      'x_c' : rect_c['x_c'],
      'y_c' : rect_c['y_c'],
      'w' : int(np.floor(rect_c['w']*scale)),
      'h' : int(np.floor(rect_c['h']*scale))
    }

def get_constrained_bounds(frame, rect_sc):
  # Make sure that the bounds are constrained to the dimensions of the frame
  #print(f"rect: {rect}")
  
  h, w, _ = frame.shape

  x_r = rect_sc['x_c']+rect_sc['w']//2
  x_l = rect_sc['x_c']-rect_sc['w']//2
  y_d = rect_sc['y_c']+rect_sc['h']//2
  y_u = rect_sc['y_c']-rect_sc['h']//2
  #print(f"w: {w} h:{h}")
  #print(f"x_r: {x_r} x_l: {x_l} y_d: {y_d} y_u: {y_u}")

  x_l_b = max(0, min(x_l, w))
  x_r_b = min(w, x_r)
  y_u_b = max(0, min(y_u, h))
  y_d_b = min(h, y_d)

  bounds_x = [x_l_b, x_r_b]
  bounds_y = [y_u_b, y_d_b]

  return bounds_x, bounds_y

def get_pdf_idx(r_bin, g_bin, b_bin, n_bins):
  return r_bin + n_bins * (g_bin + n_bins * b_bin)

def generate_bin_LUT(space_min, space_max, n_bins):
  global bin_LUT
  for v in range(space_min, space_max):
    bin_LUT[v] = int(np.floor(((v - space_min) / space_max) * n_bins))

def get_binned_vals(vals):
  binned_vals = []
  global bin_LUT
  for v in vals:
    b_val = bin_LUT[v]
    binned_vals.append(b_val)
  return binned_vals

# Formula 5
def gauss_profile_d(x):
  return 0.5*(np.exp(-0.5*x)/(2*np.pi))

def gauss_profile(x):
  return np.exp(-0.5*x)/(2*np.pi)

def get_pdf_dist_bhattacharyya(pdf_a, pdf_b):
  return np.sqrt(np.sum(np.multiply(pdf_a, pdf_b)))

# Formulas 19, 20
def get_pdf(frame, rect_c, scale=1):
  pdf = np.zeros(N_BINS**3)
  C = 0 # This is the normalization constant
  rect_sc = rect_scale(rect_c, scale)
  bounds_x, bounds_y = get_constrained_bounds(frame, rect_sc)
  for x in range(bounds_x[0], bounds_x[1]):
    for y in range(bounds_y[0], bounds_y[1]):
      # OpenCV stores images in BGR!
      b, g, r = get_binned_vals(frame[y,x,:])
      x_dist = (x-rect_sc['x_c'])**2 / rect_sc['w']
      y_dist = (y-rect_sc['y_c'])**2 / rect_sc['h']
      norm = x_dist + y_dist
      k = gauss_profile(norm)
      pdf_idx = get_pdf_idx(r, g, b, N_BINS)
      pdf[pdf_idx] = pdf[pdf_idx] + k
      C = C + k # Accumulate the normalization constant!
  return np.divide(pdf, C)

def shift_target(frame, q, p, rect_c, scale=1):
  x_shift = 0
  y_shift = 0
  C = 0 # This is the normalization constant
  rect_sc = rect_scale(rect_c, scale)
  bounds_x, bounds_y = get_constrained_bounds(frame, rect_sc)
  for x in range(bounds_x[0], bounds_x[1]):
    for y in range(bounds_y[0], bounds_y[1]):
      b, g, r = get_binned_vals(frame[y,x,:])
      # Formula 25, weight calculation
      # TODO: This should change with my changes!
      pdf_idx = get_pdf_idx(r, g, b, N_BINS)
      w = np.sqrt(q[pdf_idx]/p[pdf_idx])
      x_dist = (x-rect_sc['x_c'])**2 / rect_sc['w']
      y_dist = (y-rect_sc['y_c'])**2 / rect_sc['h']
      norm = x_dist + y_dist
      g = gauss_profile_d(norm)
      # Formula 25. This is the mean shift!
      x_shift = x_shift + x*w*g
      y_shift = y_shift + y*w*g
      C = C + w*g
  x_s = int(np.floor(x_shift/C))
  y_s = int(np.floor(y_shift/C))
  return {
    'x_c' : x_s,
    'y_c' : y_s,
    'w' : rect_sc['w'],
    'h' : rect_sc['h']
  }

def mean_shift(frame, q, rect_c, scale=1):
  # For scale variance, have the algorithm pick the best scale.
  s_min = scale-(scale*.05)
  s_max = scale+(scale*.05)
  s_step = (s_max - s_min)/8
  d_cutoff = 1
  x_found = rect_c['x_c']
  y_found = rect_c['y_c']
  scale_best = scale

  n_iterations = 0
  n_6_hits = 0

  for s in np.arange(s_min, s_max, s_step):
    x = rect_c['x_c']
    y = rect_c['y_c']
    converged = False

    while not converged:
      n_iterations += 1
      r_c = {
        'x_c' : x, 
        'y_c' : y, 
        'w' : rect_c['w'],
        'h' : rect_c['h']
      }
      p = get_pdf(frame, r_c, s)
      dist = get_pdf_dist_bhattacharyya(p, q)
      r_shifted = shift_target(frame, q, p, r_c, s)
      p_shifted = get_pdf(frame, r_shifted, s)
      dist_shifted = get_pdf_dist_bhattacharyya(p_shifted, q)

      # This is step 6 of the algorithm. Supposedly rare to encounter
      if dist_shifted < dist:
        n_6_hits += 1
        #print("Setp 6 condition!")
        #print(f" dist_shifted: {dist_shifted} < dist: {dist}")
        r_shifted = {
          'x_c' : (r_shifted['x_c'] + x)//2,
          'y_c' : (r_shifted['y_c'] + y)//2,
          'w' : r_shifted['w'],
          'h' : r_shifted['h']
        }

      # Convergence criteria (Round to same pixel)
      if r_shifted['x_c'] == x and r_shifted['y_c'] == y:
        converged = True
      else:
        x = r_shifted['x_c']
        y = r_shifted['y_c']

    # Get measures for the best scale
    p_shifted = get_pdf(frame, r_shifted, s)
    dist = get_pdf_dist_bhattacharyya(p_shifted, q)
    d = np.sqrt(1-dist)
    #print(f"d: {d}, d_cutoff: {d_cutoff}")
    if d < d_cutoff:
      scale_best = s
      d_cutoff = d
      x_found = x
      y_found = y

  #print(f"scale_best: {scale_best}")
  print(f"Converged after {n_iterations} iterations on x: {x_found}, y: {y_found} with {n_6_hits} step 6 conditions ")
  return ({
      'x_c' : x_found,
      'y_c' : y_found,
      'w' : rect_c['w'],
      'h' : rect_c['h']
    }, scale_best)

def my_method(frame, q, rect_c, scale=1):
  return (rect_c, scale)

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
  cv2.waitKey(1)

def process(sequence_path):
  seq = zipfile.ZipFile(sequence_path, 'r')

  n_frames = get_frame_count(seq)
  ground_truth = read_ground_truth(seq)

  # I found that performance is MUCH better with a lookup table for binning
  generate_bin_LUT(0, 2**BIT_DEPTH, N_BINS)

  results = {}
  results['name'] = os.path.basename(sequence_path)

  mean_shift_scale = 1
  my_method_scale = 1

  for frame_num in range(0, n_frames):
    print(f"Working on frame {frame_num}")
    # Get the frame
    frame = read_frame(seq, frame_num)
    truth_rect = ground_truth[frame_num]
    # Convert the rect to a centered rect. All functions use centered
    truth_rect_c = rect_center(truth_rect)

    if frame_num == 0:
      # Get the target_info from the initial frame
      # Just set initil bounding rectangles to the ground truth
      h, w, d = frame.shape
      print(f"Frame dimensions: ({w}, {h})")
      print(f"Frame Channels {d}")
      print(f"Target location x: {truth_rect_c['x_c']}, y: {truth_rect_c['y_c']}")
      q = get_pdf(frame, truth_rect_c)
      mean_shift_rect_c = truth_rect_c
      my_method_rect_c = truth_rect_c
    else:
      # Get the predicted bounding rectangles
      my_method_rect_c = truth_rect_c
      mean_shift_rect_c, mean_shift_scale = mean_shift(frame, q, mean_shift_rect_c, 1)
      my_method_rect_c, my_method_scale = my_method(frame, q, my_method_rect_c, 1)

    # Gather rects and uncenter them for rendering
    rects = [
      { 'rect': rect_uncenter(truth_rect_c), 'color' : (255, 0, 0) },
      { 'rect': rect_uncenter(rect_scale(mean_shift_rect_c, mean_shift_scale)), 'color' : (0, 255, 0) },
      { 'rect': rect_uncenter(rect_scale(my_method_rect_c, my_method_scale)), 'color' : (0, 0, 255) }
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
