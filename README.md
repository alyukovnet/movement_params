# Computer Vision: Movement Params

Software for calculating movement parameters of objects on video stream

## Usage

### Install

System requirements: 
`Python >=3.9`

```shell
# Clone repo to local machine
git clone https://github.com/alyukovnet/movement_params.git

# Go to downloaded directory
cd movement_params

# Create python virtual environment
python3 -m venv ./venv

# Activate virtual environment (on Linux)
source ./venv/bin/activate

# Activate virtual environment (on Windows)
.\venv\Scripts\activate

# Install pip requirements
python3 -m pip install -r ./requirements.txt
```

Put 
[cfg](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg)
and
[weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
files to `./movement_params/model` folder

### Config

In [CONFIG.py](movement_params/CONFIG.py) you can change paths of input/output 
resources and some settings.

There are two configurations - Default and Debug (based on Default)

### Run

```shell
# Run as module with default configuration
python3 -m ./movement_params

# Run as module with DEBUG configuration
python3 -m ./movement_params --debug
```

## Contributing

Use separated branches for each feature.

### Feature cycle
```shell
# Be up-to-date
git pull

# Go to main branch
git checkout main

# Move on new branch
git checkout -b 'feature/name-of-feature'

# After realisation and testing, make changed files indexed
git add .

# Commit changes
git commit -m 'Write what you did in this commit'

# Don't forget to push it on server
git push origin 'feature/name-of-feature'
```

Make pull-request on
[GitHub](https://github.com/alyukovnet/movement_params/pulls)
and wait for code review.
