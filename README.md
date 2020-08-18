# DTAM: Dense Tracking and Mapping in Real-time
## Overview
This is an implementation of [DTAM: Dense Tracking and Mapping in Real-Time - Richard A. Newcombe](http://ugweb.cs.ualberta.ca/~vis/courses/CompVis/readings/3DReconstruction/dtam.pdf). This paper proposes using all pixels instead of just some collection of feature points for tracking camera pose as well as dense mapping of environment. Merits are shown over feature based tracking as tracking is more robust to motion blur, camera defocus, even at high framerates. On the mapping side, depth of all pixels is obtained with smooth surface assumption for low-textured regions. System was designed for AR/VR application where we can first build map of the confined environment and then use this for realtime tracking with GPU.
## Motivation
Reasons for implementation:
- Better understanding of paper
- Paul Forster implemented [OpenDTAM](https://github.com/anuranbaka/OpenDTAM) under GSOC program, elaborately [logged](https://sites.google.com/a/opencv.org/opencv/gsoc-2014-project-notes#DTAM) weekly. Implementation seems rough but is well commented with citations. Further, direct GPU impl. is difficult to understand.
- impl. will further be extended for navigation purposes with coarser, trajectory relevant regions reconstruction.
## Resources
[Dense Visual SLAM - R.A. Newcombe, Phd. Thesis](https://www.doc.ic.ac.uk/~ajd/Publications/newcombe_phd2012.pdf)<span id="newcombe-thesis"></span>

[On Benchmarking Camera Calibration and Multi-View Stereo for High Resolution Imagery - C. Strecha](https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/denseMVS.html)

[super3d - Kitware impl. (looks similar to DTAM)](https://github.com/Kitware/super3d)

[Telesculptor - End to end 3d reconstruction for UAV videos (Kitware)](https://github.com/Kitware/TeleSculptor)
## Development log
> New entry is created with [snippet](.vscode/snippets/markdown.json) in vscode using prefix `log`. User defined snippets [doc](https://code.visualstudio.com/docs/editor/userdefinedsnippets).

- **03 Aug '20:**
Read [thesis](#newcombe-thesis) Ch. 1-5 Convex Optimisation Based Multi-view Stereo Depth Estimation. Need to read further about how dual formulation is derived. Revisit general primal-dual optimiztion.

- **04 Aug '20:**
Read thesis 
Ch. 6: Surface Representation, Integration and Prediction, Ch. 8: Direct Tracking from Surface Models, 9.2: DTAM: Dense Tracking and Mapping in Real-time

- **05 Aug '20:**
To begin with, we will first implement depth estimation of keyframe (5.3 Global Cost Volume Optimisation). This assumes that, we already have pose of cameras. To test, I will use the same [fountain-P11](https://github.com/openMVG/SfM_quality_evaluation/tree/master/Benchmarking_Camera_Calibration_2008/fountain-P11) dataset used by author. Camera poses for all 11 images are given. Will exhaustively search for all discrete depth at all iterations. Then incrementally, reduce search to bounds by parabola, then increase solution accuracy with subsample optimization with single Newton step. In complete framework, system is bootstrapped with poses given by PTAM till first keyframe. After that system uses dense methods using virtual camera loss.

- **06 Aug '20:**
Setup cmake project with opencv dependency. Intellisense was not working in vscode. Turns out had somehow missed popup asking *use compile_commands.json from `build` dir for intellisense* which created entry `            "compileCommands": "${workspaceFolder}/build/compile_commands.json"`. Build tested with read image. 

- **16 Aug '20:**
c++ implementation discontinued. Reasons include Eigen library is not well documented, eg. constructor for matrix from array, is it row major or column major. Debugging becomes tough for non-native types, can't visualize matrix other than raw data. Unable to get natvis file with vscode cmake debug task. natvis seems well documented for visual studio rather than vscode. Was able to successfully visualize with [python lldb script](https://github.com/tehrengruber/LLDB-Eigen-Data-Formatter). But still can't debug matrix experession like inverse. All this demanded too much attention on the impl. side than theoretical. I decide to use python instead as all of above problems are solved there. One shortcoming is ofcorse that we loose static type checking and separate static impl. will be needed for production and GPU. But pros far outweight cons as rapid prototyping is need of current time.

- **18 Aug '20:**
Photometric loss vs depth plotted for a marker point in Fountain-P11 dataset. Photometric loss along epipolar line can be noisy without clear minimum if only 1 corresponding frame is considered but, clear minimum can be seen by taking all the images in sequence with covisibility. Average has a clear local minima.
![Alt text](res/photometric_loss_vs_depth__Fountain_P11.png?raw=true "Averaged Photometric loss in sequence frames")
