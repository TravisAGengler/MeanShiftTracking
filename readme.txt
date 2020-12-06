Real-Time Tracking of Non-Rigid Ob jects using Mean Shift

scrape_dataset.py contains code to scrape the Visual Tracker Benchmark datasets from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html. This prodces a dataset.json file that can later be used in...

setup_dataset.py. This file will read dataset.json and download a specified dataset to a specified folder

mean_shift.py performs the mean shift algoritm on the specidied dataset.

run_tests.py automates downloading and running mean_shift on the datasets contained in the specified dataset.json. It will download and run all datasets