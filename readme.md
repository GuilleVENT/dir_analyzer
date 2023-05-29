## Audio Analysis Script
##### Overview
This script processes audio files in a specified directory, extracts various features using the librosa library, and stores these features in a CSV file. Optionally, the script can rename each file based on its tempo and dominant key.

Librosa is a Python library for audio and music analysis. It provides the building blocks necessary to create music information retrieval systems. It was developed by Brian McFee while at the Music and Audio Research Lab (MARL) of New York University and is now maintained by a team of developers. It is a popular library used widely in both academia and industry.

### Installation
You need Python 3.6 or later to run this script. The required Python packages can be installed via pip or Anaconda.

With pip, you can install the requirements with:

```
pip install -r requirements.txt
```

With Anaconda, you can create a new environment and install the requirements with:

```
conda env create -f environment.yml
```

### Usage 
You can run the script from the command line with:

```
python script.py [directory] [-r/--rename]
```

#### Arguments
directory (optional): The directory where the audio files are located. If not provided, the script uses the current directory.
-r or --rename (optional): If this option is specified, the script renames files after analysis according to the file's tempo and dominant key.
Output
The script creates a CSV file named audio_data.csv in the same directory, with each row representing a song and each column representing a feature.

If the rename option is used, each file is renamed to follow the pattern "Song Name [Tempo BPM] Key.extension".

Help
You can display a help message with:

```
python script.py --help
```
This will display a help message showing the usage of the script and descriptions of the arguments.

