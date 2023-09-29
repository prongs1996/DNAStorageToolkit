# Data Encoding


This directory contains implementations of three different data encoding techniques, each designed to efficiently convert data into DNA strands. These encoding methods offer varying levels of performance and capabilities.

## Table of Contents

- [Baseline Encoding](#baseline-encoding)
- [Gini Encoding](#gini-encoding)
- [DNAMapper Encoding](#dnamapper-encoding)

## Baseline Encoding

The baseline encoding implementation follows the state-of-the-art architecture described in the paper *Random access in large-scale DNA data storage* [[1]](#1). The DNA error correction uses the Reed-Solomon Error Correction Code. It leverages the State of the Art Schifra open source Library.
### Dependencies

G++

### RS encode
Set the configurations in the config file, The data/message needs to kept in the file: {input_f}
After encoding, the message with redundancy will be written to file {output_f_encoded}
```shell
python3 codec.py codec.cfg 0
```

### RS decode
Set the configurations in the config file.
The strands are specified in file {output_f_encoded}
After decoding, the original message will be written to file {output_f_decoded}

```shell
python3 codec.py codec.cfg 1
```



## Gini Encoding

As codewords tend to exhibit higher error rates in their middle indexes, the **Gini** approach adopts a different strategy by redistributing the codewords diagonally. This redistribution effectively spreads the error-prone middle indexes across all codewords uniformly. This adjustment eliminates the bias associated with the original baseline implementation, which was vulnerable to errors in specific positions. Importantly, Gini still retains the capability to correct errors caused by erasures, as a single molecule continues to be distributed across all codewords.

In the original approach, rectifying the most unreliable codewords often required multiple copies of each molecule. However, by evenly distributing this unreliability, Gini reduces the overall need for duplicating molecules to correct errors across all codewords. In simpler terms, Gini demonstrates a higher degree of error correction reliability with the same number of molecule copies.

 The implementation follows the paper [[2]](#2).


## DNAMapper Encoding


**DNAMapper** presents an alternative approach to address reliability imbalances. The concept involves categorizing different bits based on their reliability requirements, with data demanding higher reliability being mapped onto more dependable indexes. Conversely, data with lower reliability needs is mapped onto less reliable indexes. This mapping strategy is generally applicable to any data with a quality concept, such as images or videos. During the decoding process, data is handled in the usual manner, but with a crucial modification: the data from unreliable codewords now originates from bits that are more resilient to corruption. This adjustment ensures that the overall quality of retrieved images or videos is maximized, as the unreliable bits are sourced from areas that can better withstand corruption.

The DNAMapper interface is designed to be versatile and supports user-defined mapping of bits into reliability classes, allowing for customization based on specific needs and data characteristics.


The implementation follows the paper [[2]](#2).


## References
<a id="1">[1]</a> 
Organick, L., Ang, S.D., Chen, Y.J., Lopez, R., Yekhanin, S., Makarychev, K., Racz, M.Z., Kamath, G., Gopalan, P., Nguyen, B. and Takahashi, C.N., (2018). 
**Random access in large-scale DNA data storage**. 
Nature biotechnology, 36(3), pp. 242-248.


<a id="2">[2]</a> 
Lin, D., Tabatabaee, Y., Pote, Y. and Jevdjic, D., (2022, June). 
**Managing reliability skew in DNA storage**. 
In Proceedings of the 49th Annual International Symposium on Computer Architecture, pp. 482-494.
