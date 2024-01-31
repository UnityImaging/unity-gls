# Unity-GLS
## DICOM to Longitudinal Strain
Demonstration of processing an apical 4/2/3 chamber into the endocardial contours to calculate the longitudinal strain.

### 0) Create a venv
Make sure python 3.11 is already installed
```shell
python3.11 -m venv ~/venv/unity
source ~/venv/unity/bin/activate
```

### 1) Install the appropriate packages
```shell
pip3 install torch torchvision torchaudio numpy pydicom highdicom pylibjpeg[all] scipy imageio imageio-ffmpeg polarTransform requests_cache
```

### 2) Pull the repo from github
```shell
git clone https://github.com/UnityImaging/unity-gls.git
```

### 3) Run the python script
```shell
cd unity-gls
python3 ./gls_from_dicom.py --file ~/test_dicom.dcm --output_dir ~/output/
```