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
N_BINS = 32
KERNEL_CONST = np.power(2*np.pi, -3/2)

# Globals
bin_LUT = np.zeros(1)
q = np.zeros(np.power(N_BINS, 3))
p0 = np.zeros(np.power(N_BINS, 3))
p1 = np.zeros(np.power(N_BINS, 3))

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

def generate_bin_LUT(space_min, space_max, n_bins):
  global bin_LUT
  bin_LUT = np.zeros(space_max-space_min+1).astype(int)
  for v in range(space_min, space_max+1):
    bin_LUT[v] = min(n_bins-1, int(np.floor(((v - space_min) / space_max) * n_bins)))
  print(bin_LUT)

def rect_2_rad(rect):
  h0, h1 = (int(np.ceil(rect['w']/2)), int(np.ceil(rect['h']/2)))
  return {
    'x0' : rect['x'] + h0,
    'x1' : rect['y'] + h1,
    'h0' : h0,
    'h1' : h1
  }

def rad_2_rect(rad):
  return {
    'x' : rad['x0'] - rad['h0'],
    'y' : rad['x1'] - rad['h1'],
    'w' : rad['h0'] * 2,
    'h' : rad['h1'] * 2
  }

def get_g(norm_x_2):
  return 0.5*np.exp(-0.5*norm_x_2)*KERNEL_CONST

def get_k(norm_x_2):
  return np.exp(-0.5*norm_x_2)*KERNEL_CONST

def get_d(rho):
  return np.sqrt(1-rho)

def get_rho(p, q):
  return np.sum(np.sqrt(np.multiply(p, q)))

def get_norm2(x, xi, h):
  return np.linalg.norm([(xi[0]-x[0])/h[0], (xi[1]-x[1])/h[1]])

def rgb_2_u(bgr):
  b, g, r = bgr
  return bin_LUT[r] + N_BINS * (bin_LUT[g] + N_BINS * bin_LUT[b])

def round_pt(x):
  return (int(np.round(x[0])), int(np.round(x[1])))

def get_roi(x, h, img_dim):
  x1 = min(img_dim[0], x[0]+h[0])
  x0 = max(0, min(x[0]-h[0], img_dim[0]))
  y1 = min(img_dim[1], x[1]+h[1])
  y0 = max(0, min(x[1]-h[1], img_dim[1]))
  return (x0, x1), (y0, y1)

def get_hist_inplace(hist, frame, x, h):
  hist.fill(0) # Need to clear out previous values
  C = 0 # This is the normalization constant
  img_h, img_w, _ = frame.shape
  xb, yb = get_roi(x, h, (img_w, img_h))
  for xi in range(xb[0], xb[1]):
    for yi in range(yb[0], yb[1]):
      norm = get_norm2(x, (xi, yi), h)
      k = get_k(norm)
      u = rgb_2_u(frame[yi,xi,:])
      hist[u] += k
      C += k # Accumulate the normalization constant!
  hist /= C

def get_q(frame, x, h):
  global q
  get_hist_inplace(q, frame, x, h)

def get_p0(frame, y, h):
  global p0
  get_hist_inplace(p0, frame, y, h)

def get_p1(frame, y, h):
  global p1
  get_hist_inplace(p1, frame, y, h)

def mean_shift(frame, y, h0):
  y0 = y

  s_h00 = int(np.round(h0[0]*.10))
  s_h01 = int(np.round(h0[1]*.10))
  h_n10 = (h0[0]-s_h00, h0[1]-s_h01)
  h_p10 = (h0[0]+s_h00, h0[1]+s_h01)
  hs = [h_n10, h0, h_p10]

  best_d = 100
  best_h = h0
  best_y = y0

  for h in hs:
    converged = False
    while not converged:
      get_p0(frame, y0, h)
      rho0 = get_rho(p0, q)

      numx = numy = den = 0
      img_h, img_w, _ = frame.shape
      xb, yb = get_roi(y0, h, (img_w, img_h))
      for xi in range(xb[0], xb[1]):
        for yi in range(yb[0], yb[1]):
          u = rgb_2_u(frame[yi,xi,:])
          wi = np.sqrt(q[u]/p0[u])
          norm = get_norm2(y0, (xi, yi), h)
          g = get_g(norm)
          numx += xi*wi*g
          numy += yi*wi*g
          den += wi*g
      y1 = round_pt((numx/den, numy/den))

      get_p1(frame, y1, h)
      rho1 = get_rho(p1, q)

      if rho1 < rho0:
        y1 = round_pt( ((y0[0]+y1[0])/2, (y0[1]+y1[1])/2) )
        get_p1(frame, y1, h)
        rho1 = get_rho(p1, q)

      if y1 == y0:
        converged = True
        d = get_d(rho1)
        if d < best_d:
          best_y = y1
          best_h = h
          best_d = d
      else:
        y0 = y1

  return {
    'x0' : best_y[0],
    'x1' : best_y[1],
    'h0' : best_h[0],
    'h1' : best_h[1],
  }

def my_method(frame, q, y, h):
  return {
        'x0' : y[0],
        'x1' : y[1],
        'h0' : h[0],
        'h1' : h[1],
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
  cv2.waitKey(1)

def process(sequence_path):
  seq = zipfile.ZipFile(sequence_path, 'r')

  n_frames = get_frame_count(seq)
  ground_truth = read_ground_truth(seq)

  # I found that performance is MUCH better with a lookup table for binning
  generate_bin_LUT(0, 2**BIT_DEPTH, N_BINS)

  results = {}
  results['name'] = os.path.basename(sequence_path)

  for frame_num in range(0, n_frames):
    print(f"Working on frame {frame_num}")
    # Get the frame
    frame = read_frame(seq, frame_num)
    truth_rect = ground_truth[frame_num]
    # Convert the rect to a radius. All functions use centered
    truth_rad = rect_2_rad(truth_rect)

    if frame_num == 0:
      # Get the target_info from the initial frame
      # Just set initil bounding rectangles to the ground truth
      h, w, d = frame.shape
      print(f"Frame dimensions: ({w}, {h}, {d})")
      
      y0 = (truth_rad['x0'], truth_rad['x1'])
      h = (truth_rad['h0'], truth_rad['h1'])
      print(f"Target location: ({y0[0]}, {y0[1]})")
      print(f"Target dimensions: ({h[0]}, {h[1]})")
      
      get_q(frame, y0, h)
      mean_shift_rad = truth_rad
      my_method_rad = truth_rad
    else:
      # Get the predicted bounding rectangles
      y0 = (mean_shift_rad['x0'], mean_shift_rad['x1'])
      h = (mean_shift_rad['h0'], mean_shift_rad['h1'])
      print(f"Target location: ({y0[0]}, {y0[1]})")
      print(f"Target dimensions: ({h[0]}, {h[1]})")
      
      mean_shift_rad = mean_shift(frame, y0, h)
      #my_method_rad = my_method(frame, q, y0, h)
      my_method_rad = truth_rad

    # Gather rects for rendering
    rects = [
      { 'rect': rad_2_rect(truth_rad), 'color' : (255, 0, 0) },
      { 'rect': rad_2_rect(mean_shift_rad), 'color' : (0, 255, 0) },
      { 'rect': rad_2_rect(my_method_rad), 'color' : (0, 0, 255) }
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
