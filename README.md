# SPJ (Songze, Parker, Julio)
CS231N/CS230 Final Course Project

## Evaluation Code
https://github.com/ranjaykrishna/densevid_eval

## Paper and Benchmarks
https://arxiv.org/abs/1705.00754

## Dataset
http://cs.stanford.edu/people/ranjaykrishna/densevid/

## Download Instructions
Make Data directory
```
mkdir Data
cd Data
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
## ActivityNet Home Page
http://activity-net.org/

## Dense-Captioning Task Page
http://activity-net.org/challenges/2018/tasks/anet_captioning.html

## Google Cloud Instance/Image
Follow this exactly!
http://cs231n.github.io/gce-tutorial/


