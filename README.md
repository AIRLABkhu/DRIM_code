# DRIM: Depth Restoration With Interference Mitigation in Multiple LiDAR Depth Cameras

## 1. Usage

### 1.1 Setup

- Conda environment
```
conda env create -f environment.yaml
```

### 1.2 Training

Follow the instructions below to begin traning our model.

```
. run_train.sh
```

### 1.3 Testing

Follow the instructions below to begin testing our model.

The best weights are [here]([https://drive.google.com/drive/folders/1ANEa7L_j5Oz2kwvDbXlFHDLBR0aHwXF1?usp=drive_link](https://drive.google.com/file/d/1v4A00Ns-z4awYoU_eqhBWGXBLPWYGozn/view?usp=drive_link)).
```
. run_test.sh
```

Follow the instructions below to begin testing our model in challenging scenarios. When testing challenging scenarios, please change the 'test_output_dir' path in the improved_evaluate function located in utils.py.
```
. run_test_hard.sh
```
