# SPJ (Songze, Parker, Julio)
CS231N/CS230 Final Course Project

## Evaluation Code
https://github.com/ranjaykrishna/densevid_eval

## Paper and Benchmarks
https://arxiv.org/abs/1705.00754

## Dataset
http://cs.stanford.edu/people/ranjaykrishna/densevid/

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

