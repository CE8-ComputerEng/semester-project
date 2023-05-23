# Acoustic detection for warning of drowning accidents

This project is a semester project for the Computer Engineering (AI, Vison and Sound) Master's program at Aalborg University, Denmark. © 2023

## Setup
* Clone the repossitory
* Install requirements
* Move measurement files into designated folder structure as shown below
  
| Audio File                                            | Label File                                            |
|-------------------------------------------------------|-------------------------------------------------------|
| data/measurment-2/raw/230428-006.wav                  | data/measurment-2/labels/230428-006.txt              |
| data/measurment-2/raw/230428-003.wav                  | data/measurment-2/labels/230428-003.txt              |
| data/measurment-1/amplified_jumps/230320-008-jump-1.wav | data/measurment-1/labels/230320-008-jump-1.txt       |
| data/measurment-1/amplified_jumps/230320-008-jump-2.wav | data/measurment-1/labels/230320-008-jump-2.txt       |
| data/measurment-1/amplified_jumps/230320-008-jump-3.wav | data/measurment-1/labels/230320-008-jump-3.txt       |
| data/measurment-1/amplified_jumps/230320-009-jump-1.wav | data/measurment-1/labels/230320-009-jump-1.txt       |
| data/measurment-1/amplified_jumps/230320-009-jump-2.wav | data/measurment-1/labels/230320-009-jump-2.txt       |

* Run cnn_make-clips.py
* Run train_ccn.py (train_cnn.py will create numpy files that train_svm.py and train_lda.py depends on)
* If you wish to run baseline_model, then also run the make-clips.py file first.
