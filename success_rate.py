#!/usr/bin/env python3

import argparse
import json

import numpy as np

import matplotlib
matplotlib.use('agg') # Need this on windows
import matplotlib.pyplot as plt

def get_overlap_percent(truth, predict):
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

  return I / Ag

def get_true_area(truth):
  return truth['w']*truth['h']

def calc_success(threshold, rects):
  # Get the percentage of rectangles that have overlap for a given threshold
  success_orig = 0
  success_mine = 0
  total = len(rects)
  for rect in rects:
    t_area = get_true_area(rect['true'])
    overlap_orig = get_overlap_percent(rect['true'], rect['orig'])
    overlap_mine = get_overlap_percent(rect['true'], rect['mine'])
    if overlap_orig >= threshold:
      success_orig += 1
    if overlap_mine >= threshold:
      success_mine += 1

  print(success_orig)
  print(success_mine)
  return success_orig/total, success_mine/total

def main():
    results = parse_args()

    all_results_rects = []
    for result_file in results:
      with open(result_file) as f:
        result = json.load(f)
        all_results_rects.extend([r for r in result])

    overlap_threshold_x = np.arange(0.02, 1.02, 0.02)
    success_rate_orig_y = []
    success_rate_mine_y = []

    for threshold in overlap_threshold_x:
      success_rate_orig, success_rate_mine = calc_success(threshold, all_results_rects)
      success_rate_orig_y.append(success_rate_orig)
      success_rate_mine_y.append(success_rate_mine)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(overlap_threshold_x, success_rate_orig_y, color='green')
    ax.plot(overlap_threshold_x, success_rate_mine_y, color='red')
    ax.set_title('Success Plots')
    ax.set_xlabel('Overlap Threshold')
    ax.set_ylabel('Success Rate')
    fig.savefig('success_rate.png', bbox_inches='tight')
    plt.close(fig)

def parse_args():
  parser = argparse.ArgumentParser(prog="success_rate", description='Calculate success rate for given results')
  parser.add_argument('-r', '--results', type=str, nargs='+', action="append",
                  help='Paths to the rects.json for all results to be used in calculation', required=True)
  parser.add_argument('-o', '--output', metavar='PATH', type=str, nargs='?', default="success_rate.png",
                  help='The path to write the success rate plot to')
  args = vars(parser.parse_args())
  return args['results'][0]

if __name__ == '__main__':
  main()
