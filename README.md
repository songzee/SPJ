# SPJ (Songze, Parker, Julio)
CS231N/CS230 Final Course Project



## Google Cloud Instance/Image
Follow this exactly! With one exeception, storage size. Keep in mind the 'Data/' folder alone (without videos, just preprocessed features) requires about 90GB of data.
http://cs231n.github.io/gce-tutorial/



## Download Instructions
Make Data directory
```
mkdir Data
cd Data
```

Download IDs and Labels for All Splits
```
curl -O https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
unzip captions.zip
```

Download CD3 Features
```
curl -O http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-00
curl -O http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-01
curl -O http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-02
curl -O http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-03
curl -O http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-04
curl -O http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-05
curl -O http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/PCA_activitynet_v1-3.hdf5
```

Group unzipped splitted files
```
cat activitynet_v1-3.part-* > temp.zip 
```

Unzip files
```
unzip temp.zip
```


## Submission Format
```
{
version: "VERSION 1.0",
results: {
  v_5n7NCViB5TU: [
      {
      sentence: "One player moves all around the net holding the ball", # String description of an event. 
      timestamp: [1.23,4.53] # The start and end times of the event (in seconds).
      },
      {
      sentence: "A small group of men are seen running around a basketball court playing a game".
      timestamp: [5.24, 18.23]
      }
  ]
}
external_data: {
  used: true, # Boolean flag. True indicates the use of external data.
  details: "First fully-connected layer from VGG-16 pre-trained on ILSVRC-2012 training set", # This string details what kind of external data you used and how you used it.
}
}
```

## Detach Using Screen 

Start Screen Session
```
screen -S jupyter
```

start jupyter notebook
```
jupyter notebook
```

detach
Press CTRL-A, D

re-attach to this screen session
```
screen -r jupyter
```

## Adding to PYTHONPATH
```
export PYTHONPATH=$(pwd)
```

## Deep Action Proposals (DAPs Repo)
https://github.com/escorciav/daps


## NetVLAD (LOUPE Repo)
https://github.com/antoine77340/LOUPE


## Evaluation Code
https://github.com/ranjaykrishna/densevid_eval


## Quick References
Paper and Benchmarks
https://arxiv.org/abs/1705.00754

Dataset
http://cs.stanford.edu/people/ranjaykrishna/densevid/

ActivityNet Home Page
http://activity-net.org/

Dense-Captioning Task Page
http://activity-net.org/challenges/2018/tasks/anet_captioning.html
