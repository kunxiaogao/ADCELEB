import os
import sys
import pandas as pd
import shutil

def process_directory(input_dir, output_dir, df):
    spontaneous_contexts = ['interview', 'informal talk', 'public conference']
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and filename.endswith(".npy"): # Change to .wav/.txt according to your needs
            name = filename.split('_ID_')[0]
            seg = filename.split('_ID_')[1].split('_SEG_')[0]
            video_context = df.loc[(df['names'] == name) & (df['seg'] == seg), 'video_context']

            if not video_context.empty and video_context.iloc[0] in spontaneous_contexts:
                target_dir = os.path.join(output_dir, 'spontaneous')
            else:
                target_dir = os.path.join(output_dir, 'unspontaneous')
            os.makedirs(target_dir, exist_ok=True)  # Ensure the target directory exists
            shutil.copy(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} to {target_dir}")

def recursive_process(input_base_dir, output_base_dir, df):
    for root, dirs, files in os.walk(input_base_dir):
        for dir_name in dirs:
            input_sub_dir = os.path.join(root, dir_name)
            output_sub_dir = os.path.join(output_base_dir, root[len(input_base_dir):], dir_name)
            process_directory(input_sub_dir, output_sub_dir, df)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_data.py input_base_dir output_base_dir")
    else:
        input_base_dir = sys.argv[1]
        output_base_dir = sys.argv[2]
        df = pd.read_csv('')  # Path to video context info csv file
        recursive_process(input_base_dir, output_base_dir, df)
