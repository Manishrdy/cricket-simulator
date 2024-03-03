import os
import pandas as pd

folder_name = './t20/'
files = os.listdir(folder_name)

csv_filepaths = []

for i in files:
    csv_filepaths.append(folder_name+i)

data_frames = []
for csv_file_path in csv_filepaths:
  data_frame = pd.read_csv(csv_file_path)
  data_frames.append(data_frame)
  print('Merging {}.'.format(csv_file_path))

merged_data_frame = pd.concat(data_frames, ignore_index=True)

merged_csv_file_path = "t20_master.csv"
merged_data_frame.to_csv(merged_csv_file_path, index=False)