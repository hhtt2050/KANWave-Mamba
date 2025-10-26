# KANWave-Mamba
A Rice Leaf Disease Segmentation Method Based on Kolmogorov-Arnold Network and Wavelet-Enhanced State Space Model
## Data preparation

```
├── data
    ├── ALDD
        ├── ImageSets
            ├── val.txt
            ├── train.txt
        ├── JPEGImages
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── Segmentation
            ├── im1.png
            ├── im2.png
            └── ...
    ├── RLDSD
        ├── ImageSets
            ├── val.txt
            ├── train.txt
        ├── JPEGImages
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── Segmentation
            ├── im1.png
            ├── im2.png
            └── ...
```

### Prerequisites
`conda` virtual environment is recommended. 
```bash
conda create -n lsnet python=3.10
pip install -r requirements.txt
```

## Training models.
```
cd code
python train_KANWave_Mamba.py --disease RLDSD --name not_adv --root_path  /KANWve-Mamba/data  --batch_size 2  --max_iterations 80000
```
