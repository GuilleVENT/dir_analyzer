"""
Analyze audio files and optionally rename them based on the extracted features.

This script processes audio files in a specified directory (or the current directory if not specified), 
extracts various features from each audio file using librosa, and stores these features in a CSV file. 
If the rename option is specified, it also renames each file after analysis according to the file's tempo 
and dominant key. 

Usage: python script.py [directory] [-r/--rename]

Arguments:
    directory: Optional. The directory where audio files are located. If not provided, the script uses the current directory.
    -r/--rename: Optional. If specified, the script renames files after analysis. 

Librosa is used for audio analysis. Here's a brief overview of the features extracted:
    - Tempo: The estimated tempo of the song in beats per minute (BPM).
    - Dominant Key: The estimated key of the song.
    - Spectral Contrast: A measure of the difference in amplitude between peaks and valleys in a sound spectrum.
    - Spectral Centroid: Indicates where the "center of mass" of the spectrum is located.

The script also extracts the chroma feature (relates to the 12 different pitch classes) and uses it to estimate the song's key.

After extraction, features are stored in a CSV file in the same directory, with each row representing a song and each column 
representing a feature.

If the rename option is used, each file is renamed to follow the pattern "Song Name [Tempo BPM] Key.extension".
"""
import os
import librosa
import pandas as pd
import numpy as np
import re  # for regular expressions
import argparse  # for command-line arguments
import concurrent.futures

# Keys in music theory (0=C, 1=C#, 2=D, ..., 11=B)
KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def analyze_audio_files(directory, rename=False):
    """
    This function walks through a given directory and analyzes each audio file it finds.
    It extracts musical features from each file, stores them in a DataFrame, and writes the DataFrame to a CSV file.

    If the rename flag is set to True, it also renames the file based on the extracted features.

    Args:
        directory (str): The directory where audio files are located.
        rename (bool, optional): If True, the function will rename the audio files based on their features. 
                                 The renaming format is as follows: 
                                 "[original filename without any leading numbers or special characters][BPM][Key].extension". 
                                 Defaults to False.

    Returns:
        out_df (pd.DataFrame): A DataFrame containing information about each audio file, including its file path and musical features.
    """
    headers = ['Filepath', 'Dominant Key', 'Tempo', 'Spectral Contrast', 'Spectral Centroid'] + KEYS
    try:
        out_df = pd.read_csv('audio_data_v2.csv')
    except FileNotFoundError:
        out_df = pd.DataFrame(columns=headers)

    # List of files to process
    files_to_process = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp3") or file.endswith(".wav"): 
                if file not in out_df['Filepath'].tolist():
                    file_path = os.path.join(root, file)
                    files_to_process.append(file_path)

    # Use ThreadPoolExecutor to parallelize the function execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for file_path, file_data_dict in zip(files_to_process, executor.map(extract_features, files_to_process)):
            print(f'- Extracting features of: \n {file_path}')
            if file_data_dict is not None:
                print(file_data_dict)

                dict_df = pd.DataFrame(file_data_dict, index=[0])
                out_df = pd.concat([out_df, dict_df], ignore_index=True)

                if rename:
                    new_file_path = rename_file(file_path, file_data_dict)
                    out_df.replace(file_path, new_file_path, inplace=True)

        out_df.to_csv(os.path.join(directory, 'audio_data_v2.csv'), index=False)
    return out_df

def normalize_chroma(y, sr):
    """
    Normalize the chroma of an audio signal using librosa's chroma_cqt method.
    Chroma features are an interesting and powerful representation for music audio,
    whereby the entire spectrum is projected onto 12 bins representing the 12 distinct
    semitones (or chroma) of the musical octave.
    Refer to https://librosa.org/doc/main/generated/librosa.feature.chroma_cqt.html for more details.

    Args:
        y (np.array): The audio time series.
        sr (int): The sampling rate of the audio.

    Returns:
        np.array: The normalized chroma.
    """  
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return librosa.util.normalize(chroma, norm=2)

def compute_tonnetz(chroma):
    """
    Compute the tonnetz of chroma.
    Tonnetz computation is a method in musicology to represent harmonic relations.
    It considers the harmonic relations between different pitches in a chroma vector.
    Refer to https://librosa.org/doc/main/generated/librosa.feature.tonnetz.html for more details.

    Args:
        chroma (np.array): Chroma of an audio signal.

    Returns:
        np.array: Mean of tonnetz.
    """   
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    return np.mean(tonnetz, axis=1)

def find_key(chroma):
    """
    Find the dominant key of an audio signal from the normalized chroma.
    It calculates the mean across time for each pitch class, and returns the pitch
    class with the highest mean value, which is considered the dominant key.

    Args:
        chroma (np.array): Normalized chroma of an audio signal.

    Returns:
        str: The dominant key.
    """
    return KEYS[np.argmax(np.mean(chroma, axis=1))]

def extract_features(file_path):
    """
    Extract audio features from a file.
    This function uses librosa's load function to load an audio file.
    It extracts the tempo, beat_frames, chroma, tonnetz, spectral_contrast and spectral_centroid
    from the audio signal.
    Refer to https://librosa.org/doc/main/feature.html for more details on these features.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        dict: A dictionary of features if extraction was successful, None otherwise.
    """
    try:
        # load audio file with librosa
        y, sr = librosa.load(file_path)

        # Extracting features
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        chroma = normalize_chroma(y, sr)
        tonnetz = compute_tonnetz(chroma)
        key = find_key(chroma)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Create a data dictionary
        data_dict = {"Filepath": file_path, "Dominant Key": key, "Tempo": round(tempo,2), 
                     "Spectral Contrast": round(spectral_contrast,3), "Spectral Centroid": round(spectral_centroid,3)}
        # Include the estimations for all keys
        for k, t in zip(KEYS, np.mean(chroma, axis=1)):
            data_dict[k] = round(t,3)

        return data_dict

    except Exception as e:
        print(f"Error occurred for {file_path} : {str(e)}")
        return None

def rename_file(old_filepath, file_data_dict):
    """
    Rename the audio file based on the extracted features.

    Args:
        old_filepath (str): The old file path of the audio file.
        file_data_dict (dict): The dictionary containing the extracted features.

    Returns:
        str: The new (or old if error) path
    """
    try: 
        dirname = os.path.dirname(old_filepath)
        old_filename = os.path.basename(old_filepath)
        filename_without_ext, file_extension = os.path.splitext(old_filename)

        # Check if the filename already contains the pattern "_[{BPM}]_{KEY}_"
        if not re.search(r"_\[\d+\.\d+\]_[A-G](#|b)?_", filename_without_ext):
        
            # Use regular expressions to process the filename
            cleaned_filename = re.sub(r'^[^a-zA-Z]*', '', filename_without_ext)  # Remove any leading non-alpha characters
            cleaned_filename = re.sub(r'\(.*\)|\[.*\]|www\..*\.com|from YouTube|original|Original|', '', cleaned_filename)  # Remove anything in () or [] or any www.***.com or 'from YouTube'

            # Remove any sequence that matches the pattern of a YouTube video ID
            #cleaned_filename = re.sub(r'[0-9A-Za-z_-]{11}', '', cleaned_filename)

            # Construct new filename with BPM and Key
            new_filename = f"{cleaned_filename}_[{file_data_dict['Tempo']}]_{file_data_dict['Dominant Key']}_{file_extension}"

            # Construct full new filepath and rename the file
            new_filepath = os.path.join(dirname, new_filename)
            os.rename(old_filepath, new_filepath)

            return new_file_path
        return old_filepath
    except Exception as e:
        print(f"Error occurred while renaming file {old_filepath} : {str(e)}")
        return old_filepath

def save_to_csv(df, directory):
    """
    Save the DataFrame to a CSV file.

    This function saves the DataFrame containing audio feature data to a CSV file in the specified directory. The CSV file is named 'audio_data_v2.csv'.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved. It should contain the audio feature data extracted from the audio files.
        directory (str): The directory where the CSV file will be saved.

    Returns:
        None
    """
    csv_path = os.path.join(directory, 'audio_data_v2.csv')
    print(csv_path)

    # Sort DataFrame based on Key and Tempo
    #df = df.sort_values(by=["Dominant Key", "Tempo"], ascending=[True, True])

    df.to_csv(csv_path, index=False)


def main():
    """
    Main function to analyze audio files and optionally rename them based on the extracted features.
    It requires a directory path as command line argument and a flag -r/--rename to rename files after analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze audio files and optionally rename them based on the extracted features.")
    parser.add_argument('directory', nargs='?', default=os.getcwd(), help="Directory containing the audio files. If not provided, the current directory is used.")
    parser.add_argument('-r', '--rename', action='store_true', help="Rename files after analysis.")
    args = parser.parse_args()

    data = analyze_audio_files(args.directory, args.rename)
    data.to_csv(os.path.join(args.directory, 'audio_data_v2.csv'), index=False)

if __name__ == "__main__":
    main()
