#!/usr/bin/env python3

import argparse
# import distutils
# import errno
# import hashlib
# import os
# import pprint
# import re
import subprocess
import json
import sys
# import shlex
# import shutil

# from pathlib import Path
# from distutils import dir_util

def windowize_path(path):
  if sys.platform.startswith("win32"):
    return path.replace('/','\\')
  return path

def run_cmd(cmd, *args):
  cmd_arr = [windowize_path(cmd)]
  for arg in args:
    if isinstance(arg, list):
      for a in arg:
        cmd_arr.append(windowize_path(a))
    else:
      cmd_arr.append(windowize_path(arg))
  try:
    out = subprocess.check_output(cmd_arr, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as e:
      print(e.output.decode('utf-8'))
      return "ERROR"
  return str(out, encoding='utf-8')

def get_sequences(dataset):
  seuqnce_names = []
  with open(dataset) as f:
    dataset = json.load(f)
    for s in dataset:
      seuqnce_names.append(s['name'])
  return seuqnce_names

def main():
  dataset = parse_args()

  for s in get_sequences(dataset):
    print(f"Running sequence {s}...")
    ret = run_cmd("python", "./setup_dataset.py", "-d", "dataset.json", "-s", f"{s}", "-o", "data")
    if ret == "ERROR":
      print(f"Failed to setup sequence {s}")
      continue
    ret = run_cmd("python", "./mean_shift.py", "-s", f"data/{s}.zip")
    if ret == "ERROR":
      print(f"Failed to run mean_shift on sequence {s}")
      continue

  print("Finished")

def parse_args():
  parser = argparse.ArgumentParser(prog="run_tests", description='Run the specified sequences after downloading')
  parser.add_argument('-d', '--dataset', metavar='PATH', type=str, nargs='?', default="",
                  help='The xml animation file to convert', required=True)
  args = vars(parser.parse_args())
  return args['dataset']

if __name__ == '__main__':
  main()