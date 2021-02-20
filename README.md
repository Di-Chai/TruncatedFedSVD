## Truncated FedSVD

#### Prepare the Env

```bash
pip install -r requirements.txt
```

#### Run Trials

```bash
# -d : dataset (Currently only works with synthetic data, i.e., load_synthetic)
# -k : topk principle component
# -f : number of features,
# -p : number of parties
# -s : number of samples per data holder
# -b : number of building blocks
python TruncatedFedSVD.py -d load_synthetic -k 2 -f 100000 -p 2 -s 5000 -b 100
```

#### Example Output

Using `python TruncatedFedSVD.py -d load_synthetic -k 2 -f 100000 -p 2 -s 5000 -b 100`

```bash
PCA mode, subtracting the mean
Namespace(block=100, dataset='load_synthetic', log_dir='', num_feature=100000, num_participants=2, num_pc=2, num_samples=5000, only_time='False', output_pkl='False')
StandalonePCA time 611.3196821212769
StandalonePCA explained var ratio [0.00017315 0.00017281] 
Generate orthogonal matrix done. Using 26.462397813796997 seconds.
Apply distortion done. Using 457.81944477558136 seconds.
SVD done. Using 526.4050447940826 seconds.
Truncated FedSVD explained var ratio [0.00017316 0.00017276] 
MAPE to normal PCA 5.933551973188348e-10 
Truncated FedSVD totally uses 1010.686887383461 seconds
```