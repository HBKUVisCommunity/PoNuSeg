# PoNuSeg

**PoNuSeg: Point-Supervised Nuclei Instance Segmentation via Boundary-Aware Affinity and Contrastive Neighborhood Learning**

## Overview

PoNuSeg is a point-supervised framework for nuclei instance segmentation in histopathology images.  
The method learns instance-level nuclei masks from nucleus center annotations only, reducing the need for expensive dense pixel-wise labels.

The framework follows a two-stage design:

- **Stage I:** Generates pseudo supervision from point annotations using K-means-based foreground discovery, seed-guided partitioning, uncertainty-band construction, and pseudo-label refinement.
- **Stage II:** Trains a segmentation network to predict nuclei regions together with 8-neighborhood affinity maps and per-pixel embeddings, using reliability-aware segmentation, affinity, and contrastive objectives.

## Repository Status

This repository is currently being prepared for public release.

**The code will be made publicly available after the acceptance of the paper.**

## Paper

**Title:**  
*PoNuSeg: Point-Supervised Nuclei Instance Segmentation via Boundary-Aware Affinity Maps and Contrastive Neighborhood Learning*

## Code Availability

The implementation will be released in this repository upon paper acceptance.

Repository link:  
[https://github.com/HBKUVisCommunity/PoNuSeg.git](https://github.com/HBKUVisCommunity/PoNuSeg.git)

## Planned Contents

The public release is expected to include:

- Training and evaluation code
- Pseudo-label generation pipeline
- Refinement stage implementation
- Inference and instance decoding scripts
- Documentation for reproducing the reported experiments

## Citation

If you find this work useful, please cite the paper once it becomes available.

```bibtex
@article{ponuseg,
  title={PoNuSeg: Point-Supervised Nuclei Instance Segmentation via Boundary-Aware Affinity Maps and Contrastive Neighborhood Learning},
  author={Althelaya, Khaled and Agus, Marco and Schneider, Jens},
  journal={TBD},
  year={TBD}
}
