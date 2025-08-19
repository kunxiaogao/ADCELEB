# ADCeleb: A Longitudinal Speech Dataset from Public Figures for Early Detection of Alzheimer’s Disease
## Overview📈
This work investigates novel speech‐based approaches for Alzheimer’s disease (AD) detection, directly tackling the common lack of pre‐diagnosis samples in existing pathological speech collections. To fill this gap, we introduce ADCeleb, a unique longitudinal corpus comprising recordings from 40 celebrities diagnosed with AD alongside 40 matched control speakers. Covering a period from ten years before diagnosis up to the year of diagnosis, ADCeleb offers an unprecedented window into how subtle speech markers evolve as AD develops.
## ADCeleb
The ADCeleb corpus itself lives on [Zenodo repository](https://zenodo.org/records/15515841) as a metadata-only package—containing YouTube URLs, speaker profiles, and transcription files, but no raw audio. To work with the recordings, you pull code from our GitHub repo: its scripts download each subject’s audio, cut the files into segments using the provided timestamps, and compute both linguistic and acoustic features. During processing, the audio is automatically sorted into per-speaker directories and clipped to the exact time spans indicated in the metadata.

## 1. Installation️💻
To set up the project locally, follow these steps:

  1. Clone the Repository:
  
    git clone https://github.com/kunxiaogao/ADCELEB.git
    cd ADCELEB

  2. Install Dependencies:

  Create a virtual environment and install the required packages:

    pip install -r requirements2.txt

## 2. Get Audio Files 🎧🎵
After downloading the Zenodo repository, you can download the audio files for each speaker using the provided script. The script takes the root Zenodo directory as a parameter, which contains the metadata files with YouTube links. To download the audio files inside each speaker's folder, follow these steps:
  1. Navigate to the project directory.
  2. Run the command below, specifying the root directory of the Zenodo dataset. This script will use the ```metadata.csv``` files in each speaker’s folder to download the corresponding YouTube videos as audio files.
     
    python data/download/download_audios.py --root_dir path_to_zenodo_directory
    
  3. The script generate_speakers_folders.py reads the speaker-timestamp CSV for a video and, for each unique speaker ID (e.g. SPEAKER_OO), makes a dedicated folder. It then slices out each audio snippet defined by that speaker’s start–end times and writes it as a WAV file named after the segment’s end timestamp (e.g. 12345.wav). To create these per-speaker audio directories for a given video, use:
    
    python data/download/generate_speakers_folders.py path_to_zenodo_directory
  4. The preprocessing folder contains three scripts:

  - `transcribe.py` converts WAV files into plain-text transcripts in TXT format. These transcripts can be used to derive linguistic features.
  - `video_context_info_total.py` reads all `speakers_info.csv` files and produces a video-level context summary CSV (aggregating per-video metadata across speakers/segments).
  - `spontaneous_split.py` uses the CSV file produced by `video_context_info_total.py` to split audio segments into spontaneous vs non-spontaneous subsets, writing them to corresponding directories.
  5. Extract Features: The scripts to extract the acoustic features and linguistic features are located in:

  ```
  /features_extraction/acoustic_embeddings/
  ```
  ```
  /features_extraction/linguistic_embeddings/
  ```

## ⚠️Important Note on Data Availability and Disk Space Requirements
### Data Availability
Some of the URLs in the metadata may have become unreachable—whether through removal, new access restrictions, or regional blocking. All links were verified when the dataset was assembled, but online availability can change. We recommend checking each link’s accessibility before attempting any downloads.
### Disk Space and File Formats
Managing this dataset locally can demand a lot of storage, especially if you’re working with high-fidelity audio or video. Here’s an illustrative breakdown:
- **Choose compressed formats**：

  Using codecs like Opus or AAC/M4A can dramatically shrink your download size. For instance, you might end up with roughly 13 GB of compressed audio, plus another 9 GB for any concatenated, post-processed files.
- **Beware of partial availability**：

  Geo-blocking, removals, or other restrictions mean you may not be able to retrieve every clip listed in the metadata.
- **Watch for re-encoding artifacts**：

  Although recompression saves space, it can introduce minor audio quality losses during steps like concatenation, which in turn could subtly affect downstream feature extraction.

### Recommendations
  1. Allocate sufficient disk space based on the anticipated size of the dataset.
  2. Verify data availability early to account for geo-blocking or other access limitations.
  3. Consider the trade-off between compression and quality, depending on your project's requirements.










