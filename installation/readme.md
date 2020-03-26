# Medical Zoo Pytorch Installation Guide


## Set up a Virtual enviroment
```
python3 -m venv env_zoo
source env_zoo/bin/activate
pip install -r requirements.txt
```

## Set up a Conda Virtual enviroment
```
conda create -n env_zoo python=3.6
source activate env_zoo
pip install -r requirements.txt
```

## Set up a Docker image
```
1) open the terminal and change directory to the project folder
2) run the following command: sudo docker build -t enviroment:latest .
3) to make sure the docker image is created execute: docker images
4) run in sudo mode: docker run -it enviroment
```


## Where to find the supported datasets

### iSeg-2017
```
http://iseg2017.web.unc.edu/
```
### MR-BRAINS 2018 DATASET
```
https://mrbrains18.isi.uu.nl/
```

