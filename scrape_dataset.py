#!/usr/bin/env python3

import argparse
import json
import re
import requests 

BASE_URL = 'http://cvlab.hanyang.ac.kr/tracker_benchmark'
DATA_URL = BASE_URL + '/datasets.html'

def load_page(url):
  print(f"Reading dataset from {url}")
  rsp = requests.get(url)
  return rsp.content.decode('utf-8')

def get_data_name(z):
  name = re.search('seq\/(.*).zip', z)
  return name.group(1)

def get_data_url(z):
  return BASE_URL + '/' + z

def get_attribs(a):
  with_spaces = a[a.find('>')+1:a.rfind('<')].strip().split(',')
  return [s.strip() for s in with_spaces]

def get_dataset(page_data):
  found_zips = re.findall("seq\/.*zip", page_data)
  found_attribs = re.findall("<small>\s*.*\s*<\/small>", page_data)
  if len(found_zips) != len(found_attribs):
    print("Mismatch in number of datasets and attributes")
    exit()
  dataset = []
  for i in range(0, len(found_zips)):
    z = found_zips[i]
    a = found_attribs[i]
    data = {}
    data['name'] = get_data_name(z)
    data['url'] = get_data_url(z)
    data['attribs'] = get_attribs(a)
    dataset.append(data)
  return dataset

def write_dataset(dataset, out_file):
  with open(out_file, "w") as out:
    json.dump(dataset, out, indent=2, sort_keys=True)

def main():
  page_data = load_page(DATA_URL)
  dataset = get_dataset(page_data)
  write_dataset(dataset, 'dataset.json')

if __name__ == '__main__':
  main()
