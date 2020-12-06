#!/usr/bin/env python3

import argparse
import json
import os
import urllib.request

from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
  if os.path.exists(output_path):
    print(f"{output_path} already downloaded. Skipping")
  else:
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
      urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def get_requested_sequences(dataset, sequences):
  requested_seqs = []
  with open(dataset) as f:
    dataset = json.load(f)
    for s in dataset:
      if s['name'] in sequences:
        requested_seqs.append(s)
  return requested_seqs

def download_sequences(sequences, out_dir):
  for s in sequences:
    download_url(s['url'], os.path.join(out_dir, s['name'] + '.zip'))

def main():
  dataset, seq, out_dir = parse_args()
  print(f"Using dataset: {dataset}")
  print(f"Downloading sequences: {seq}")
  if not os.path.exists(out_dir):
    print(f"Creating directory {out_dir}")
    os.mkdir(out_dir)
  sequences = get_requested_sequences(dataset, seq)
  print(sequences)
  download_sequences(sequences, out_dir)

def parse_args():
  parser = argparse.ArgumentParser(prog="setup_datasets", description='Download the specified sequences')
  parser.add_argument('-d', '--dataset', metavar='PATH', type=str, nargs='?', default="",
                  help='The xml animation file to convert', required=True)
  parser.add_argument('-s', '--sequences', metavar='PATH', type=str, nargs='+', default="",
                  help='The sequences to download', required=True)
  parser.add_argument('-o', '--output', metavar='PATH', type=str, nargs='?', default="data",
                  help='The directory to download sequences to')
  args = vars(parser.parse_args())
  return args['dataset'], args['sequences'], args['output']

if __name__ == '__main__':
  main()
