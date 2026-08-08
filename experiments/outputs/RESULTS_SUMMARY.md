# Experiment summary (10 seeds unless noted)


### TKPI missing-rate sweep
```
 rate      knn_k10       knn_k5   softimpute   masked_nmf   missforest   crossover
    5 0.916±0.064 0.930±0.070 0.984±0.041 1.032±0.055 0.944±0.071   BEST=knn_k10
   10 0.926±0.028 0.948±0.041 0.981±0.048 1.008±0.052 0.949±0.063   BEST=knn_k10
   15 0.944±0.028 0.969±0.020 0.971±0.048 1.018±0.057 0.959±0.033   BEST=knn_k10
   20 0.958±0.009 0.983±0.018 0.982±0.049 1.003±0.028 0.978±0.028   BEST=knn_k10
   25 0.987±0.011 1.003±0.016 0.981±0.028 1.015±0.033 0.987±0.025   BEST=softimpute
   30 1.025±0.014 1.045±0.018 0.989±0.014 1.032±0.024 1.005±0.025   BEST=softimpute
   35 1.044±0.025 1.061±0.025 0.999±0.016 1.054±0.032 1.012±0.029   BEST=softimpute
   40 1.081±0.030 1.104±0.031 1.014±0.019 1.068±0.048 1.056±0.090   BEST=softimpute
   45 1.115±0.047 1.132±0.046 1.016±0.017 1.091±0.054 1.035±0.038   BEST=softimpute
   50 1.145±0.044 1.161±0.045 1.045±0.043 1.134±0.100 1.060±0.072   BEST=softimpute
```
- crossover knn_k10 -> softimpute at 25% missing

### USDA MCAR sweep
```
 rate      knn_k10       knn_k5   softimpute   crossover
    5 0.631±0.058 0.607±0.034 0.805±0.042   BEST=knn_k5
   10 0.721±0.025 0.716±0.029 0.832±0.026   BEST=knn_k5
   15 0.783±0.022 0.787±0.025 0.853±0.018   BEST=knn_k10
   20 0.850±0.017 0.861±0.016 0.867±0.018   BEST=knn_k10
   25 0.911±0.022 0.924±0.021 0.882±0.022   BEST=softimpute
   30 0.966±0.014 0.977±0.014 0.888±0.022   BEST=softimpute
   35 1.006±0.017 1.019±0.014 0.893±0.016   BEST=softimpute
   40 1.051±0.017 1.067±0.019 0.918±0.012   BEST=softimpute
   45 1.070±0.018 1.084±0.020 0.945±0.016   BEST=softimpute
   50 1.080±0.021 1.096±0.021 0.970±0.023   BEST=softimpute
```
- crossover knn_k5 -> knn_k10 at 15% missing
- crossover knn_k10 -> softimpute at 25% missing

### USDA MAR sweep
```
 rate      knn_k10       knn_k5   softimpute   crossover
    5 0.641±0.025 0.635±0.022 0.821±0.015   BEST=knn_k5
   10 0.704±0.037 0.688±0.034 0.836±0.026   BEST=knn_k5
   15 0.762±0.028 0.761±0.036 0.845±0.029   BEST=knn_k5
   20 0.792±0.020 0.799±0.021 0.869±0.025   BEST=knn_k10
   25 0.844±0.020 0.854±0.027 0.884±0.016   BEST=knn_k10
   30 0.900±0.018 0.914±0.015 0.894±0.016   BEST=softimpute
   35 0.936±0.018 0.954±0.015 0.926±0.013   BEST=softimpute
   40 0.982±0.024 0.996±0.028 0.939±0.012   BEST=softimpute
   45 1.017±0.019 1.034±0.021 0.951±0.017   BEST=softimpute
   50 1.043±0.014 1.062±0.015 0.970±0.018   BEST=softimpute
```
- crossover knn_k5 -> knn_k10 at 20% missing
- crossover knn_k10 -> softimpute at 30% missing

### USDA MNAR sweep
```
 rate      knn_k10       knn_k5   softimpute   crossover
    5 1.511±0.291 1.545±0.463 2.480±0.482   BEST=knn_k10
   10 1.310±0.185 1.316±0.192 2.346±0.343   BEST=knn_k10
   15 1.310±0.143 1.395±0.149 2.571±0.398   BEST=knn_k10
   20 1.648±0.169 1.773±0.188 2.540±0.317   BEST=knn_k10
   25 1.992±0.237 2.080±0.228 2.406±0.300   BEST=knn_k10
   30 2.732±0.468 2.924±0.476 2.503±0.285   BEST=softimpute
   35 3.000±0.333 3.117±0.366 2.426±0.178   BEST=softimpute
   40 3.200±0.315 3.335±0.334 2.364±0.171   BEST=softimpute
   45 3.149±0.247 3.217±0.254 2.380±0.243   BEST=softimpute
   50 3.138±0.340 3.191±0.350 2.278±0.380   BEST=softimpute
```
- crossover knn_k10 -> softimpute at 30% missing

### Scenario A (20% random) main results
```
  knn_k10          NRMSE=0.9584±0.0090   MAE=173.75±14.13
  mice_extratrees  NRMSE=0.9722±0.0337   MAE=191.79±17.05
  missforest       NRMSE=0.9776±0.0283   MAE=193.48±18.22
  softimpute       NRMSE=0.9803±0.0492   MAE=198.93±14.61
  knn_k5           NRMSE=0.9835±0.0181   MAE=178.90±14.43
  mean             NRMSE=1.0024±0.0016   MAE=214.87±11.90
  masked_nmf       NRMSE=1.0027±0.0283   MAE=195.03±10.34
  knn_k3           NRMSE=1.0063±0.0207   MAE=183.92±14.77
  median           NRMSE=1.0490±0.0070   MAE=156.17±17.76
  mice             NRMSE=1.0536±0.0545   MAE=220.09±21.48
```

### Scenario B (block micro) main results
```
  mice             NRMSE=0.9688±0.0207   MAE=271.55±28.25
  knn_k10          NRMSE=0.9919±0.0148   MAE=243.37±25.50
  softimpute       NRMSE=0.9997±0.0265   MAE=241.88±20.05
  mean             NRMSE=1.0027±0.0019   MAE=284.50±8.68
  knn_k5           NRMSE=1.0203±0.0265   MAE=244.25±27.96
  masked_nmf       NRMSE=1.0236±0.0221   MAE=262.59±15.21
  median           NRMSE=1.0404±0.0084   MAE=205.51±18.27
  missforest       NRMSE=1.0470±0.0377   MAE=313.53±33.43
  knn_k3           NRMSE=1.0494±0.0444   MAE=247.87±27.90
  mice_extratrees  NRMSE=1.2537±0.1720   MAE=364.51±32.09
```

### Additional baselines, Scenario A
```
  dae            NRMSE=0.9744±0.0209
  iterativesvd   NRMSE=1.0150±0.0618
```

### Wilcoxon KNN k=10 vs SoftImpute, n=10
```
Wilcoxon signed-rank (Scenario A, n=10 seeds)
KNN k=10 median NRMSE per seed: [0.9565, 0.9632, 0.9669, 0.9551, 0.956, 0.9436, 0.958, 0.9756, 0.9603, 0.9488]
SoftImpute median NRMSE per seed: [1.0021, 0.9451, 0.9983, 0.931, 0.961, 0.972, 0.9744, 1.0939, 1.003, 0.922]
mean KNN=0.9584  mean SI=0.9803
W=12.000  p=0.1309
```

### Downstream classification + covariance, USDA
```
oracle_clf_acc=0.8719
               nrmse  clf_acc  cov_err      prec_err  rank_nrmse  rank_acc  rank_cov
method                                                                              
missforest    0.6591   0.8160   0.8205  7.814970e+01         1.0       1.0       1.0
knn_k10       0.7787   0.7961   1.3242  2.655962e+02         2.0       2.0       4.0
mice          0.8416   0.7723   0.8994  7.627762e+02         3.0       5.0       2.0
softimpute    0.8688   0.7740   1.2369  9.268530e+01         4.0       3.0       3.0
iterativesvd  0.9232   0.7736   1.4192  7.688620e+01         5.0       4.0       5.0
masked_nmf    0.9948   0.3743   4.4463  3.741637e+06         6.0       8.0       8.0
mean          1.0003   0.7648   1.8631  2.735084e+02         7.0       7.0       7.0
median        1.0250   0.7684   1.8457  2.740372e+02         8.0       6.0       6.0
```

### Cold-start scaling regimes
```
  oracle   median NRMSE=1.0717   #<1.0=3/19
  donor    median NRMSE=1.0717   #<1.0=3/19
  naive    median NRMSE=27.9632   #<1.0=0/19
```
oracle = held-out column scaled by its own observed range (the main-pipeline setting)
donor  = scaled by USDA range for same nutrient (external side info)
naive  = pooled range of other columns (no per-column scale prior)