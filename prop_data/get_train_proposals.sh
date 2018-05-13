#!/bin/bash
# ```
#
# Sample bash-script used to read all video ids from csv files
# and run Proposal Module to return proposals into csv files
#
#
# Usage: ./get_proposals.sh OR sh get_proposals.sh
# Requirements: conda
#
# ```

DIR="../daps"
MODEL="T512K64_thumos14.npz"
DIR_C3D="../c3d_data"
VideoID="v_sJFgo9H6zNo"
while read VideoID; do
  echo "VideoID: $VideoID"
  $DIR/tools/generate_proposals.py -iv $VideoID -ic3d $DIR_C3D/sub_activitynet_v1-3.c3d.hdf5 -imd $DIR/$MODEL
  echo "\n"
done <train_ids.csv