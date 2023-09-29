# Wetlab Simulation

This directory features two distinct simulators for wetlab steps of DNA synthesis, storage, PCR amplification and DNA sequencing. 

## Table of Contents

- [Naive Wetlab Simulator](#naive-wetlab-simulator)
- [RNN Simulator](#rnn-simulator)

## Naive Wetlab Simulator

The **Naive Wetlab Simulator** is a straightforward simulator that replicates DNA synthesis and sequencing processes using basic modeling. This implementation follows Rashtchian et al. [[1]](#1).

### Usage
```
python3 /naive/noise.py --N coverage  --subs P_sub --dels P_del --inss P_ins  --i EncodedStrands.txt  --o UnderlyingClusters.txt
python3 /naive/shuffle.py UnderlyingClusters.txt  NoisyStrands.txt
```

## RNN Simulator

The **RNN Simulator** adopts a more advanced approach by employing an RNN with GRUs to simulate DNA synthesis and sequencing processes. This simulator leverages machine learning techniques to provide a more sophisticated and accurate simulation of wetlab activities.

## References
<a id="1">[1]</a> 
Rashtchian, C., Makarychev, K., Racz, M., Ang, S., Jevdjic, D., Yekhanin, S., Ceze, L. and Strauss, K., (2017). **Clustering billions of reads for DNA data storage**. Advances in Neural Information Processing Systems, 30.