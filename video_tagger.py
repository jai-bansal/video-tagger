# This script creates a video tagger for videos in '.mp4' format.
# First, videos are split into frames.
# Then, image recognition using open-source networks is
# applied to each frame.
# The resulting tags are then further processed.

# NOTE: The image recognition models are initialized with
# weights derived from training them on the ImageNet dataset
# (as opposed to random initialization weights). The weights
# for some of these models are >100 MB and so are NOT included.
# They must be downloaded into the 'weights' folder for the code to work.
# The links where these weights files can be found are below.

################
# IMPORT MODULES
################
import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50, xception, vgg19, inception_v3, inception_resnet_v2, mobilenet, nasnet, imagenet_utils

##################
# CREATE FUNCTIONS
##################
# This section creates functions to be used below.

# Create function that takes predictions and adds resulting counts
# to a dictionary.
def update_dict(preds, result_dict, prob_cutoff = 0.25):

  # Loop through 'preds'.
  for i in range(len(preds[0])):

    # If the returned tag IS in 'result_dict', add 1 to the associated count.
    # Only add categories with probability greater than some threshold.
    if preds[0][i][1] in result_dict and preds[0][i][2] >= prob_cutoff:
      result_dict[preds[0][i][1]] += 1
   
    # If the returned tag is NOT in 'result_dict', add it.
    # Only add categories with probability greater than some threshold.
    if preds[0][i][1] not in result_dict and preds[0][i][2] >= prob_cutoff:
      result_dict[preds[0][i][1]] = 1

  # Return updated 'result_dict'.
  return(result_dict)

# Create function to remove dictionary entries (tags) if they didn't
# appear frequently enough.
def filter_dict(dictionary, percentage = 0.05):

  # Return updated 'dictionary'
  return({key: value for key, value in dictionary.items() if value > int((frame_count / 30) * percentage)})

#########################
# SPLIT VIDEO INTO FRAMES
#########################

# Import video (needs full path for some reason...)
vid = cv2.VideoCapture('small.mp4')

# Get first frame of 'vid' and create frame counter variable.
valid_input, frame = vid.read()
frame_count = 0
print('Splitting video into frames')

# Go through video frames and save as images.
while valid_input:                              # Run as long as there are valid frames.

  # Get next frame.
  valid_input, frame = vid.read()

  # Save frames as JPEG images every so often.
  # These are saved into the 'frames' folder.
  if frame_count % 30 == 0:
      cv2.imwrite('frames/frame%d.jpg' % frame_count, frame)

  # Update 'frame_count'.
  frame_count += 1

print('Done Splitting')
print('')

#################################
# CREATE IMAGE RECOGNITION MODELS
#################################
# This section creates image recognition model objects.
# These models need to use weights derived from training
# on the ImageNet data set (as opposed to random weights,
# which would probably suck).
# These weights should be saved in the 'weights' folder.

# Unfortunately, creating models objects with the 'weights = 'imagenet' argument
# results in a long download which usually fails.
# Instead, I recommend pre-downloading the weights locally.
# The relevant files can be found at:
# Resnet50: https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
# Xception: https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5
# VGG19: https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5
# InceptionV3: https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
# InceptionResNetV2: https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5
# MobileNet: https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5
# NASNetLarge: https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-large.h5

# Create models (make sure weights are already downloaded!).
# This takes a while...
print('Loading Models')
resnet = resnet50.ResNet50(weights = 'weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
xc = xception.Xception(weights = 'weights/xception_weights_tf_dim_ordering_tf_kernels.h5')
v19 = vgg19.VGG19(weights = 'weights/vgg19_weights_tf_dim_ordering_tf_kernels.h5')
ic3 = inception_v3.InceptionV3(weights = 'weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
ic_resnet = inception_resnet_v2.InceptionResNetV2(weights = 'weights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
mobile = mobilenet.MobileNet(weights = 'weights/mobilenet_1_0_224_tf.h5')
nn_large = nasnet.NASNetLarge(weights = 'weights/NASNet-large.h5')
print('Models Loaded')
print('')

#############################################
# RUN FRAMES THROUGH IMAGE RECOGNITION MODELS
#############################################

# Create dictionary to hold results for each model.
resnet_results = dict()
xc_results = dict()
v19_results = dict()
ic3_results = dict()
ic_resnet_results = dict()
mobile_results = dict()
nn_large_results = dict()

# Loop through frames.
# If the tagger is run for multiple videos, the 'frames' folder should be emptied
# between videos!
print('Generating Results')
print('')
for frame in os.listdir('frames'):
  if frame.endswith('.jpg'):                                 # Only consider JPG files.

    # Load frame. Different models prefer different sizes.
    im_224 = image.load_img('frames/' + frame,
                            target_size = (224, 224))
    im_299 = image.load_img('frames/' + frame,
                            target_size = (299, 299))
    im_331 = image.load_img('frames/' + frame,
                            target_size = (331, 331))

    # More image pre-processing.
    im_224 = image.img_to_array(im_224)
    im_224 = np.expand_dims(im_224, axis = 0)       # This gives the image 4 dimensions and is necessary for future steps.

    im_299 = image.img_to_array(im_299)
    im_299 = np.expand_dims(im_299, axis = 0)              

    im_331 = image.img_to_array(im_331)
    im_331 = np.expand_dims(im_331, axis = 0)

    # Generate predictions for each model.
    resnet_pred = imagenet_utils.decode_predictions(resnet.predict(resnet50.preprocess_input(im_224)), 5)
    xc_pred = imagenet_utils.decode_predictions(xc.predict(xception.preprocess_input(im_299)), 5)
    v19_pred = imagenet_utils.decode_predictions(v19.predict(vgg19.preprocess_input(im_224)), 5)
    ic3_pred = imagenet_utils.decode_predictions(ic3.predict(inception_v3.preprocess_input(im_299)), 5)
    ic_resnet_pred = imagenet_utils.decode_predictions(ic_resnet.predict(inception_resnet_v2.preprocess_input(im_299)), 5)
    mobile_pred = imagenet_utils.decode_predictions(mobile.predict(mobilenet.preprocess_input(im_224)), 5)
    nn_large_pred = imagenet_utils.decode_predictions(nn_large.predict(nasnet.preprocess_input(im_331)), 5)

    # Update result dictionaries based on tags.
    resnet_results = update_dict(preds = resnet_pred, result_dict = resnet_results, prob_cutoff = 0.25)
    xc_results = update_dict(preds = xc_pred, result_dict = xc_results, prob_cutoff = 0.25)
    v19_results = update_dict(preds = v19_pred, result_dict = v19_results, prob_cutoff = 0.25)
    ic3_results = update_dict(preds = ic3_pred, result_dict = ic3_results, prob_cutoff = 0.25)
    ic_resnet_results = update_dict(preds = ic_resnet_pred, result_dict = ic_resnet_results, prob_cutoff = 0.25)
    mobile_results = update_dict(preds = mobile_pred, result_dict = mobile_results, prob_cutoff = 0.25)
    nn_large_results = update_dict(preds = nn_large_pred, result_dict = nn_large_results, prob_cutoff = 0.25)

# Remove tags that appear too infrequently.
resnet_results = filter_dict(resnet_results)
xc_results = filter_dict(xc_results)
v19_results = filter_dict(v19_results)
ic3_results = filter_dict(ic3_results)
ic_resnet_results = filter_dict(ic_resnet_results)
mobile_results = filter_dict(mobile_results)
nn_large_results = filter_dict(nn_large_results)

# Print results.
print('Resnet: ', sorted(resnet_results, key = resnet_results.get, reverse = True))
print('Xception: ', sorted(xc_results, key = xc_results.get, reverse = True))
print('VGG19: ', sorted(v19_results, key = v19_results.get, reverse = True))
print('InceptionV3: ', sorted(ic3_results, key = ic3_results.get, reverse = True))
print('Inception Resnet: ', sorted(ic_resnet_results, key = ic_resnet_results.get, reverse = True))
print('Mobile: ', sorted(mobile_results, key = mobile_results.get, reverse = True))
print('NASNet: ', sorted(nn_large_results, key = nn_large_results.get, reverse = True))


  


        
      
