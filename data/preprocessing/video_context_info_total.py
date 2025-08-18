import os
import pandas as pd

# Base directory where the group folders are located
base_dir = ''

# Initialize a list to store data
data = []

# Walk through the directories
for group in os.listdir(base_dir):  # Group level, e.g., 'AD'
    group_path = os.path.join(base_dir, group)
    if os.path.isdir(group_path):
        for name in os.listdir(group_path):  # Name level
            name_path = os.path.join(group_path, name)
            if os.path.isdir(name_path):
                for seg in os.listdir(name_path):  # Segment level, e.g., '2Ud1VvmN1j8'
                    seg_path = os.path.join(name_path, seg)
                    speakers_info_path = os.path.join(seg_path, 'speakers_info.csv')
                    if os.path.isfile(speakers_info_path):
                        # Read the CSV file
                        df = pd.read_csv(speakers_info_path)
                        # Extract the video context from the row with the 'target' status
                        video_context = df.loc[df['status'] == 'target', 'video_context'].values[0]
                        # Append the information to the data list
                        data.append({
                            'group': group,
                            'names': name,
                            'seg': seg,
                            'video_context': video_context
                        })

# Create the final DataFrame
final_df = pd.DataFrame(data)

# Print or save the DataFrame
print(final_df)
final_df.to_csv('', index=False)

unique_video_contexts = final_df['video_context'].unique()

# Print the unique video contexts
print("Unique video contexts:")
print(unique_video_contexts)
