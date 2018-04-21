# SPJ (Songze, Parker, Julio)
CS231N/CS230 Final Course Project


# Dense Captioning Events in Video - Evaluation Code

## Usage
First, clone this repository and make sure that all the submodules are also cloned properly.
```
git clone --recursive https://github.com/ranjaykrishna/densevid_eval.git
```

Next, download the dataset using
```
./download.sh
```

Finally, test that the evaluator runs by testing it on our ```sample_submission.json``` file by calling:
```
python evaluate.py
```

You are now all set to produce your own dense captioning results for videos and use this code to evaluate your mode:
```
python evaluate.py -s YOUR_SUBMISSION_FILE.JSON
```
