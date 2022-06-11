# final-motion
The final repository for HAR fairness tests, using the modtion sense dataset.

Federated learning files use flower simulation over 24 participants. Each FL file turns data into binary file, references these during training. 

Currently stuck on DP + CL, as DP appears to reduce accuracy greatly 96 -> 69.

bianryCNN.py contains the model without DP, dpCNN.py contains the model with DP.

part_windows_folder and preprocessed_folder are folders for binary files.

This project requires the archive folder from the motion-sense dataset to run.
