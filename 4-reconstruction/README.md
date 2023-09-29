# Trace Reconstruction

This directory houses implementations of three distinct trace reconstruction algorithms, each designed to reconstruct encoded DNA strands from clusters of sequenced noisy DNA strands. 

## Table of Contents

- [Single-Sided BMA Reconstruction](#single-sided-bma-reconstruction)
- [Double-Sided BMA Reconstruction](#double-sided-bma-reconstruction)
- [Needleman-Wunsch Reconstruction](#needleman-wunsch-reconstruction)

## Single-Sided BMA Reconstruction

The baseline implementation or the **Single-Sided BMA Reconstruction** algorithm follows the BMA-lookahead algorithm for DNA-based data storage as proposed by Organick et al. [[1]](#1).

### Usage
```
python3 4-reconstruction/recon.py --i input_file --o output_file --L strand_length --W window_size --coverage COVERAGE --ALG 1
```

## Double-Sided BMA Reconstruction

The **Double-Sided BMA Reconstruction** algorithm first reconstructs the left half of the consensus strand from left to right using the BMA-lookahead algorithm from left to right, and then reconstructs the right half of the consensus strand by using the BMA-lookahead algorithm from right to left. The two halves are then joined to create the final consensus strand. Our implementation follows the paper [[2]](#2).

### Usage
```
python3 4-reconstruction/recon.py --i input_file --o output_file --L strand_length --W window_size --coverage COVERAGE --ALG 0
```

## Needleman-Wunsch Reconstruction

The **Needleman-Wunsch Reconstruction** computes the multiple sequence alignment of all noisy strands within a cluster by using the Needleman-Wunsch algorithm [[3]](#3) [[4]](#4) to compute the global alignments that minimizes edit distance. Once it has this alignment, the algorithm can reconstruct the consensus strand by merely taking the majority vote at every index.


### Usage
```
python3 4-reconstruction/recon.py --i input_file --o output_file --L strand_length --coverage COVERAGE --ALG 2
```

## References
<a id="1">[1]</a> 
Organick, L., Ang, S.D., Chen, Y.J., Lopez, R., Yekhanin, S., Makarychev, K., Racz, M.Z., Kamath, G., Gopalan, P., Nguyen, B. and Takahashi, C.N., (2018). 
**Random access in large-scale DNA data storage**. 
Nature biotechnology, 36(3), pp. 242-248.


<a id="2">[2]</a> 
Lin, D., Tabatabaee, Y., Pote, Y. and Jevdjic, D., (2022, June). 
**Managing reliability skew in DNA storage**. 
In Proceedings of the 49th Annual International Symposium on Computer Architecture, pp. 482-494.


<a id="3">[3]</a> 
Christopher Lee, (2003).
**Generating consensus sequences from partial order multiple sequence alignment graphs.**
Bioinformatics, 19(8):999–1008.


<a id="4">[4]</a> 
Christopher Lee, Catherine Grasso, and Mark F Sharlow, (2002).
**Multiple sequence alignment using partial order graphs.**
Bioinformatics, 18(3): 452–464.