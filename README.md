#### Synopsis:
This project builds a simple video tagger. Relevant topics/labels are returned for any (well, sort of) video.

#### Motivation:
I wrote this project to experiment with open source image recognition models (more below) and some video analysis.

#### Method:
Step 1: break a video up into frames
Step 2: run image recognition on each frame
Step 3: process outputs (for example, remove tags that appear infrequently) and return results

#### Notes:
- A short video is provided in the repo.
- Only every 30th frame is used (corresponding to 30 FPS) to keep things manageable.
- A variety of open source image recognition models are tried.
- IMPORTANT: The image recognition models are initialized with weights derived from training them on the ImageNet dataset (as opposed to random initialization weights). The weights for some of these models are >100 MB and so are NOT included. They must be downloaded into the 'weights' folder for the code to work. The links where these weights files can be found are included in the script comments.
