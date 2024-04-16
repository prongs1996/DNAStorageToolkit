## DNA Storage Toolkit: A Modular End-to-End DNA Data Storage Codec and Simulator

With the exponential growth in data generation, the need for efficient and cost-effective data storage solutions has never been greater. Advances in DNA synthesis and sequencing technologies have sparked interest in using DNA as a medium for long-term, high-density data storage. However, despite numerous proposals for DNA storage architectures, there is a lack of comprehensive tools for research in this field.

This repository provides an open-source, end-to-end DNA data storage toolkit that facilitates every step of the DNA-based data storage pipeline. Our toolkit offers implementations of cutting-edge techniques, including custom algorithms, for each phase of the pipeline. These phases encompass data encoding into DNA strands, simulation of wetlab processes for synthesis, storage, and sequencing of DNA strands, clustering of sequenced results, reconstruction of DNA strands from noisy clusters, and decoding the initially encoded file. Researchers can utilize each module independently or combine them to construct a complete DNA storage pipeline.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [License](#license)

## Features

- **End-to-End DNA Data Storage:** Our toolkit covers the entire DNA-based data storage pipeline, from encoding data into DNA to decoding it.
- **Modular Design:** Each phase of the pipeline is implemented as a separate module, allowing for flexibility and easy integration into custom workflows.
- **State-of-the-Art Algorithms:** We provide implementations of the latest techniques in DNA data storage, as well as our own algorithms for specific pipeline components.
- **Error Correction:** Support for error-correction mechanisms ensures data integrity throughout the pipeline.
- **Open Source:** Our toolkit is open-source and free to use, encouraging collaboration and innovation in the field of DNA data storage.

## Installation

To get started with our DNA storage pipeline toolkit, you need to install the following:
> * Python 3
> * G++
> * pyspoa
> * editdistance

For the RNN simulator, you will need:
> * matplotlib
> * tqdm
> * mlconfig
> * numpy
> * pandas
> * Pillow
> * torch
> * torchvision



After installing these, you can simply clone this repo and use the modules directly:

```
$ git clone https://github.com/DNAStorageToolkit/DNAStorageToolkit.git

$ cd DNAStorageToolkit
```
To see the pipeline in action, run the demo made available directly:
```
$ sh demo1.sh
```


## Usage

To demonstrate how to use the toolkit, we have provided a demo script ([**demo1.sh**](./demo1.sh)) that takes an image through the entire DNA storage pipeline.

For further info on how to use each module independently, check the readme files in the individual directories.

## Modules

Our pipeline consists of the following modules:

1. **Data Encoding:** Convert your data into DNA sequences. 
*directory*: [1-encoding-decoding](./1-encoding-decoding/)
2. **Wetlab Simulation:** Simulate the processes of DNA synthesis, storage, and sequencing. 
*directory*: [2-simulating_wetlab](./2-simulating_wetlab/)
3. **Clustering:** Group sequenced DNA strands into clusters for efficient processing. 
*directory*: [3-clustering](./3-clustering/)
4. **Reconstruction:** Reconstruct original DNA strands from noisy clusters. 
*directory*: [4-reconstruction](./4-reconstruction/)
5. **Decoding:** Decode DNA strands back into the original data with support for error correction. 
*directory*: [1-encoding-decoding](./1-encoding-decoding/)

Each module is designed to be used independently or in combination, allowing you to tailor your DNA storage pipeline to your specific needs.


## Citation

If you use this toolkit, please cite our paper:
```
@inproceedings{sharma2024dnatoolkit,
  title={DNA Storage Toolkit: A Modular End-to-End DNA Data Storage Codec and Simulator},
  author={Sharma, Puru and Goh Yipeng, Gary and Gao, Bin and Ou, Longshen and Lin, Dehui and Sharma, Deepak and Jevdjic, Djordje},
  booktitle={2024 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  year={2024},
  abbr={ISPASS'24},
  organization={IEEE}
}
```


We hope that our DNA storage pipeline toolkit proves valuable to storage researchers and developers who are exploring the exciting possibilities of DNA-based data storage. If you have any questions or feedback, please don't hesitate to reach out. 🧬📦🚀
