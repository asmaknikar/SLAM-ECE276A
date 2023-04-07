# ECE276A-Project 1- Panorama Generation and Orientation Tracking

We need the following folder structure to run our code
```

├── code
│   ├── figures
│   └── pickles
|   README.md
├── data
│   ├── testset
│   │   ├── cam
│   │   └── imu
│   └── trainset
│       ├── cam
│       ├── imu
│       └── vicon
├── docs
```
To generate the images in the figures folder and pickles in the pickles folder,
* change `MODE = "train"` for training mode in line 301 of `test.py`
* change `MODE = "test"` for test mode in line 301 of `panorama.py`


follwed by running the following in a the terminal:


```
cd ./code
python test.py
```

To generate the panorama in the images folder 
* change `MODE = "train"` for training mode in line 52 of `panorama.py`
* change `MODE = "test"` for test mode in line 52 of `panorama.py`


follwed by running the following in a the terminal:

```
cd ./code
python panorama.py
```

The locations of the datasets should be self explanatory from folder structure. But one can change their locations by changing the varibales after the import statements
