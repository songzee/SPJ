filename = 'Data/sub_activitynet_v1-3.c3d.hdf5'
fid = h5py.File(filename, 'r')
video_lst = list(fid.keys())
print('Number of videos in the container: ', len(video_lst))
ith = 5
print('feat of {}-th video: {}'.format(ith, video_lst[ith]))
# This line clearly shows that the features are stored as Group/Dataset
feat_video_ith = fid[video_lst[ith]]['c3d_features'][:]
print('shape of features: {}'.format(feat_video_ith.shape))
print('It should be a 2D array with the second axis == 500 for all the videos')

print('Note that the video_lst comes from a ".keys()", thus the order of might change after every trial. However, I did not noticed that in my version of python and h5py.')
fid.close()
