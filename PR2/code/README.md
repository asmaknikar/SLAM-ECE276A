# ECE276A-Project 2- Particle Filter Slam

We need the following folder structure to run our code
```

├── code
│   ├── images
|   |--README.md
├── data
│   ├── dataRBGD
│   │   ├── 
│   │   └── 
│   └── Encoders20.npz
│   └── ....npz
├── docs
```
To change the dataset change the dataset global variable in `deadReckoning.py` and `particleFilter.py`,

follwed by running the following in a the terminal:


```
cd ./code
python deadReckoning.py
python particleFilter.py

```

The locations of the datasets and output images should be self explanatory from folder structure. But one can change their locations by changing the varibales after the import statements
