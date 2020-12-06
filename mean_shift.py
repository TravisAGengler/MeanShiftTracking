#!/usr/bin/env python3

import argparse
import cv2
import io
import json
import numpy as np
import os
import re
import time
import zipfile

from datetime import datetime
from scipy.spatial import distance

import matplotlib
matplotlib.use('agg') # Need this on windows
import matplotlib.pyplot as plt

# Constants
BIT_DEPTH = 8
FRAMERATE = 15
N_BINS = 32
KERNEL_CONST = np.power(2*np.pi, -3/2)
EPSILON = np.finfo(float).eps
MAX_ITERATIONS = 25

# Globals
# These are here for performance.
bin_LUT = np.zeros(1)
q = np.zeros(np.power(N_BINS, 3))
p0 = np.zeros(np.power(N_BINS, 3))
p1 = np.zeros(np.power(N_BINS, 3))

q_mine = np.zeros(np.power(N_BINS, 3))
p0_mine = np.zeros(np.power(N_BINS, 3))
p1_mine = np.zeros(np.power(N_BINS, 3))

def write_results(results, results_dir):
  print(f"Writing results to: {results_dir}")
  timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
  result_out_dir = os.path.join(results_dir, timestamp + '-' + os.path.splitext(results['name'])[0])
  if not os.path.exists(results_dir):
    print(f"Creating output root directory {results_dir}")
    os.mkdir(results_dir)
  if not os.path.exists(result_out_dir):
    print(f'Creating output directory {result_out_dir}')
    os.mkdir(result_out_dir)
  os.chdir(result_out_dir)

  # Write the video out
  frame_h, frame_w, _ = results['frames'][0]['frame_with_rects'].shape
  out = cv2.VideoWriter('marked_video.avi',cv2.VideoWriter_fourcc(*"MJPG"), FRAMERATE, (frame_w,frame_h))
  for frame in results['frames']:
    frame_img = frame['frame_with_rects']
    out.write(frame_img)
  out.release()

  # Write the plots out
  frames_x = range(1, len(results['frames'])+1)

  fig, ax = plt.subplots(nrows=1, ncols=1)
  iou_orig_y = [iou['orig']['iou'] for iou in results['frames']]
  iou_mine_y = [iou['mine']['iou'] for iou in results['frames']]
  ax.plot(frames_x, iou_orig_y, color='green')
  ax.plot(frames_x, iou_mine_y, color='red')
  ax.set_title('Intersection / Union (Higher is better)')
  ax.set_xlabel('Frame #')
  ax.set_ylabel('IoU')
  fig.savefig('iou.png', bbox_inches='tight')
  plt.close(fig)

  fig, ax = plt.subplots(nrows=1, ncols=1)
  iters_orig_y = [it['orig']['iters'] for it in results['frames']]
  iters_mine_y = [it['mine']['iters'] for it in results['frames']]
  ax.plot(frames_x, iters_orig_y, color='green')
  ax.plot(frames_x, iters_mine_y, color='red')
  ax.set_title('Iterations till convergence (Lower is better)')
  ax.set_xlabel('Frame #')
  ax.set_ylabel('Iterations')
  fig.savefig('iters.png', bbox_inches='tight')
  plt.close(fig)

  fig, ax = plt.subplots(nrows=1, ncols=1)
  iters_orig_y = [t['orig']['time'] for t in results['frames']]
  iters_mine_y = [t['mine']['time'] for t in results['frames']]
  ax.plot(frames_x, iters_orig_y, color='green')
  ax.plot(frames_x, iters_mine_y, color='red')
  ax.set_title('Time to converge (seconds) (Lower is better)')
  ax.set_xlabel('Frame #')
  ax.set_ylabel('Time (s)')
  fig.savefig('time.png', bbox_inches='tight')
  plt.close(fig)


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

# Got good better performance with a Look up table for the bins
def generate_bin_LUT(space_min, space_max, n_bins):
  global bin_LUT
  bin_LUT = np.zeros(space_max-space_min+1).astype(int)
  for v in range(space_min, space_max+1):
    bin_LUT[v] = min(n_bins-1, int(np.floor(((v - space_min) / space_max) * n_bins)))

# Convert a rectange into the point/radius form expected in mean_shift
def rect_2_rad(rect):
  h0, h1 = (int(np.ceil(rect['w']/2)), int(np.ceil(rect['h']/2)))
  return {
    'x0' : rect['x'] + h0,
    'x1' : rect['y'] + h1,
    'h0' : h0,
    'h1' : h1
  }

# Convert a point/radius into a rectange expected for display
def rad_2_rect(rad):
  return {
    'x' : rad['x0'] - rad['h0'],
    'y' : rad['x1'] - rad['h1'],
    'w' : rad['h0'] * 2,
    'h' : rad['h1'] * 2
  }

# Find the norm of a vector
# This is a HUGE performance hog as well.
def get_norm2(x, xi, h):
  return np.linalg.norm([(xi[0]-x[0])/h[0], (xi[1]-x[1])/h[1]])

# Look up u (pixel value) for the given RGB
# This is bgr because that is how OpenCV stores colors!
def rgb_2_u(bgr):
  b, g, r = bgr
  return bin_LUT[r] + N_BINS * (bin_LUT[g] + N_BINS * bin_LUT[b])

# Round a point to the nearest pixel
def round_pt(x):
  return (int(np.round(x[0])), int(np.round(x[1])))

# Constrain the region of interest to the frame
def get_roi(x, h, img_dim):
  x1 = min(img_dim[0], x[0]+h[0])
  x0 = max(0, min(x[0]-h[0], img_dim[0]))
  y1 = min(img_dim[1], x[1]+h[1])
  y0 = max(0, min(x[1]-h[1], img_dim[1]))
  return (x0, x1), (y0, y1)

# This is the Generalized Intersection / Union
# See https://giou.stanford.edu/
def get_giou(truth, predict):
  Bg = [truth['x'], truth['y'], 
        truth['x'] + truth['w'], truth['y'] + truth['h']]
  Bp = [predict['x'], predict['y'], 
        predict['x'] + predict['w'], predict['y'] + predict['h']]

  # This ensures winding order of Bp
  x1hp = min(Bp[0], Bp[2])
  x2hp = max(Bp[0], Bp[2])
  y1hp = min(Bp[1], Bp[3])
  y2hp = max(Bp[1], Bp[3])

  Ag = (Bg[2] - Bg[0])*(Bg[3] - Bg[1])
  Ap = (x2hp - x1hp)*(y2hp - y1hp)

  x1I = max(x1hp, Bg[0])
  x2I = min(x2hp, Bg[2])
  y1I = max(y1hp, Bg[1])
  y2I = min(y2hp, Bg[3])

  I = 0
  if x2I > x1I and y2I > y1I:
    I = (x2I-x1I)*(y2I-y1I)

  x1c = min(x1hp, Bg[0])
  x2c = max(x2hp, Bg[2])
  y1c = min(y1hp, Bg[1])
  y2c = max(y2hp, Bg[3])

  Ac = (x2c - x1c)*(y2c - y1c)

  U = Ap + Ag - I
  IoU = I / float(U)

  GIoU = IoU - ((Ac - U)/float(Ac))

  return GIoU

# These functions are the papers method ----------------------------

# Derivative of Formula 4
def get_g(norm_x_2):
  if norm_x_2 < 1:
    return 0.75
  if norm_x_2 == 1:
    return 0.375
  return EPSILON

# Formula 4
def get_k(norm_x_2):
  if norm_x_2 < 1:
    return 0.75*(1-norm_x_2)
  return EPSILON

# Formula 17
def get_rho(p, q):
  return np.sum(np.sqrt(np.multiply(p, q)))

# Formula 18
def get_d(rho):
  return np.sqrt(1-rho)

# This generates the probability density function for a ROI
# Formulas 19, 21
def get_hist_inplace(hist, frame, x, h):
  hist.fill(0) # Need to clear out previous values
  C = EPSILON # This is the normalization constant
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

# This is the mean shift algorithm described in the paper
# While the paper makes claims that it achieves 30 fps, I cannot achieve such performance
# Python profiler points to issues with the nested for loops
def mean_shift(frame, y, h0):
  start_time = time.time()
  y0 = y

  # This is the scale variation described in 4.3
  s_h = round_pt((h0[0]*.10, h0[1]*.10))
  h_n10 = (h0[0]-s_h[0], h0[1]-s_h[1])
  h_p10 = (h0[0]+s_h[0], h0[1]+s_h[1])
  hs = [h_n10, h0, h_p10]

  best_d = 100
  best_h = h0
  best_y = y0
  n_iterations = 0

  for h in hs:
    converged = False
    while not converged:
      n_iterations += 1
      # Step 1
      get_p0(frame, y0, h)
      rho0 = get_rho(p0, q)

      # Step 2 and 3
      numx = numy = den = EPSILON
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

      # Step 3
      get_p1(frame, y1, h)
      rho1 = get_rho(p1, q)

      # Step 4
      if rho1 < rho0:
        y1 = round_pt( ((y0[0]+y1[0])/2, (y0[1]+y1[1])/2) )
        get_p1(frame, y1, h)
        rho1 = get_rho(p1, q)

      # Step 5
      if y1 == y0 or n_iterations > MAX_ITERATIONS:
        converged = True
        d = get_d(rho1)
        if d < best_d:
          best_y = y1
          best_h = h
          best_d = d
      else:
        y0 = y1

  print(f"Converged after {n_iterations} iterations")
  return ({
    'x0' : best_y[0],
    'x1' : best_y[1],
    'h0' : best_h[0],
    'h1' : best_h[1],
  }, {
    'iterations' : n_iterations,
    'time' : time.time() - start_time
  })

# These functions are my method -----------------------------------

def get_g_gauss(norm_x_2):
  if norm_x_2 < 1:
    return 0.75
  if norm_x_2 == 1:
    return 0.375
  return EPSILON
  #return 0.5*np.exp(-0.5*norm_x_2)*KERNEL_CONST

def get_k_gauss(norm_x_2):
  if norm_x_2 < 1:
    return 0.75*(1-norm_x_2)
  return EPSILON
  #return np.exp(-0.5*norm_x_2)*KERNEL_CONST

def get_rho_jensen_shannon(p, q):
  #return np.sum(np.sqrt(np.multiply(p, q)))
  return distance.jensenshannon(p,q)

def get_d_jensen_shannon(rho):
  #return np.sqrt(1-rho)
  return np.sqrt(rho)

def get_hist_inplace_mine(hist, frame, x, h):
  hist.fill(0) # Need to clear out previous values
  C = EPSILON # This is the normalization constant
  img_h, img_w, _ = frame.shape
  xb, yb = get_roi(x, h, (img_w, img_h))
  for xi in range(xb[0], xb[1]):
    for yi in range(yb[0], yb[1]):
      norm = get_norm2(x, (xi, yi), h)
      k = get_k_gauss(norm)
      u = rgb_2_u(frame[yi,xi,:])
      hist[u] += k
      C += k # Accumulate the normalization constant!
  hist /= C

def get_q_mine(frame, x, h):
  global q_mine
  get_hist_inplace(q_mine, frame, x, h)

def get_p0_mine(frame, y, h):
  global p0_mine
  get_hist_inplace(p0_mine, frame, y, h)

def get_p1_mine(frame, y, h):
  global p1_mine
  get_hist_inplace(p1_mine, frame, y, h)

# This method is identical to the mean_shift method with a few notable changes:
# The rho calculation is done with Jenson-Shannon instead of Bhattacharyya
# The distance measure is updated to accomodate for the change
# We are still using the epanechnikov kernel for calculating histograms and gradient
def my_method(frame, y, h0):
  start_time = time.time()
  y0 = y

  s_h = round_pt((h0[0]*.10, h0[1]*.10))
  h_n10 = (h0[0]-s_h[0], h0[1]-s_h[1])
  h_p10 = (h0[0]+s_h[0], h0[1]+s_h[1])
  hs = [h_n10, h0, h_p10]

  best_d = 100
  best_h = h0
  best_y = y0
  n_iterations = 0

  for h in hs:
    converged = False
    while not converged:
      n_iterations += 1
      get_p0_mine(frame, y0, h)
      rho0 = get_rho_jensen_shannon(p0_mine, q)

      numx = numy = den = EPSILON
      img_h, img_w, _ = frame.shape
      xb, yb = get_roi(y0, h, (img_w, img_h))
      for xi in range(xb[0], xb[1]):
        for yi in range(yb[0], yb[1]):
          u = rgb_2_u(frame[yi,xi,:])
          wi = np.sqrt(q_mine[u]/p0_mine[u])
          norm = get_norm2(y0, (xi, yi), h)
          g = get_g_gauss(norm)
          numx += xi*wi*g
          numy += yi*wi*g
          den += wi*g
      y1 = round_pt((numx/den, numy/den))

      get_p1_mine(frame, y1, h)
      rho1 = get_rho_jensen_shannon(p1_mine, q_mine)

      if rho1 < rho0:
        y1 = round_pt( ((y0[0]+y1[0])/2, (y0[1]+y1[1])/2) )
        get_p1_mine(frame, y1, h)
        rho1 = get_rho_jensen_shannon(p1_mine, q_mine)

      if y1 == y0 or n_iterations > MAX_ITERATIONS:
        converged = True
        d = get_d_jensen_shannon(rho1)
        if d < best_d:
          best_y = y1
          best_h = h
          best_d = d
      else:
        y0 = y1

  print(f"Converged after {n_iterations} iterations")
  return ({
    'x0' : best_y[0],
    'x1' : best_y[1],
    'h0' : best_h[0],
    'h1' : best_h[1],
  }, {
    'iterations' : n_iterations,
    'time' : time.time() - start_time
  })

# Convert status and other info into a result that we will write to disk
def get_frame_results(truth_rect, rects, frame_rects, mean_shift_stats, my_method_stats):
  return {
    'frame_with_rects' : frame_rects,
    'orig': {
      'iou' : get_giou(truth_rect['rect'], rects[0]['rect']),
      'iters' : mean_shift_stats['iterations'],
      'time' : mean_shift_stats['time']
    },
    'mine': {
      'iou' : get_giou(truth_rect['rect'], rects[1]['rect']),
      'iters' : my_method_stats['iterations'],
      'time' : my_method_stats['time']
    }
  }

# Draw rectangles on a frame
def draw_rects(img, rects):
  for rect in rects:
    r = rect['rect']
    tl = (r['x'], r['y'])
    br = (r['x'] + r['w'], r['y'] + r['h'])
    color = rect['color']
    thickness = 2
    rect_img = cv2.rectangle(img, tl, br, color, thickness)
  return rect_img

# Display the frame with rectangles.
def display_results(frame, rects):
  frame_rects = draw_rects(frame, rects)
  cv2.imshow('',frame_rects)
  cv2.waitKey(1)
  return frame_rects

def process(sequence_path):
  seq = zipfile.ZipFile(sequence_path, 'r')

  n_frames = get_frame_count(seq)
  ground_truth = read_ground_truth(seq)

  # I found that performance is MUCH better with a lookup table for binning
  generate_bin_LUT(0, 2**BIT_DEPTH, N_BINS)

  results = {}
  results['name'] = os.path.basename(sequence_path)
  results['frames'] = []

  for frame_num in range(0, n_frames):
    print(f"Working on frame {frame_num}")
    # Get the frame
    frame = read_frame(seq, frame_num)
    truth_rect = ground_truth[frame_num]
    # Convert the rect to a radius. All functions use centered
    truth_rad = rect_2_rad(truth_rect)

    if frame_num == 0:
      # Get the target_info from the initial frame
      # Just set initial bounding rectangles to the ground truth
      h, w, d = frame.shape
      print(f"Frame dimensions: ({w}, {h}, {d})")
      
      y0 = (truth_rad['x0'], truth_rad['x1'])
      h = (truth_rad['h0'], truth_rad['h1'])
      y0_mine = (truth_rad['x0'], truth_rad['x1'])
      h_mine = (truth_rad['h0'], truth_rad['h1'])

      print(f"Target location: ({y0[0]}, {y0[1]})")
      print(f"Target dimensions: ({h[0]}, {h[1]})")
      
      get_q(frame, y0, h)
      get_q_mine(frame, y0_mine, h_mine)
      mean_shift_rad = truth_rad
      my_method_rad = truth_rad
      mean_shift_stats = {
        'iterations' : 0,
        'time': 0
      }
      my_method_stats = {
        'iterations' : 0,
        'time': 0
      }
    else:
      # Get the predicted bounding rectangles
      y0 = (mean_shift_rad['x0'], mean_shift_rad['x1'])
      h = (mean_shift_rad['h0'], mean_shift_rad['h1'])
      y0_mine = (my_method_rad['x0'], my_method_rad['x1'])
      h_mine = (my_method_rad['h0'], my_method_rad['h1'])

      mean_shift_rad, mean_shift_stats = mean_shift(frame, y0, h)
      my_method_rad, my_method_stats = my_method(frame, y0_mine, h_mine)
      #my_method_rad = truth_rad
      #mean_shift_rad = truth_rad

    # Gather rects for rendering
    rects = [
      { 'rect': rad_2_rect(truth_rad), 'color' : (255, 0, 0) },
      { 'rect': rad_2_rect(mean_shift_rad), 'color' : (0, 255, 0) },
      { 'rect': rad_2_rect(my_method_rad), 'color' : (0, 0, 255) }
    ]

    # Show frames here if we want to!
    frame_rects = display_results(frame, rects)

    # Do the analysis here, accumulate into results
    res = get_frame_results(rects[0], rects[1:], frame_rects, mean_shift_stats, my_method_stats)
    results['frames'].append(res)

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
