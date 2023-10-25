# Single-cell-challenge
![](https://norbis.w.uib.no/files/2016/05/F1.large_-768x623.jpg](https://data.prod.sagebase.org/3324230/57c5394d-5a59-4610-89da-10a758a73f89/Single%20Cell%20Transcriptomics%20Banner.png?response-content-disposition=attachment%3B+filename%3D%22Single+Cell+Transcriptomics+Banner.png%22%3B+filename*%3Dutf-8%27%27Single%2520Cell%2520Transcriptomics%2520Banner.png&response-content-type=image%2Fpng&X-Amz-Date=20231025T153914Z&X-Amz-Expires=30&Expires=1698248384&Signature=DPzVzRjcocGoWv9PqqEPrhC9Er6XOMhbodz1kFPGtNqRAMoVVSwGWVtCwwEcwWvMWcy5v5k7PZxWv3yNcPIkxcYDqJTrnIBCN01OFLsyI3D2Aj7VEki8M5sd3SR7BecbEQJktQmVxh19dROsO1oYQ2OEMr5BGjmv6IcvbQAm2NYwGokp02HsAfP4GdoTK~cziryz~lQabvWrD~6rElnDK883q7YzqyfIFWT75tsUdIzosotsWm4~2W33C6BkPeMRv6u6xQPB4-kO-VVZz6ZmwAF-YnMu8bhHAjfEk7z1OmsSZ8OdzJqX7EZ~By8bM1g6zdjJnuv9K49S5QtovgsEWQ__&Key-Pair-Id=K9FM1UY7AFDBB)

## Overview
Single-cell sequencing technologies are rapidly evolving. In particular, although suspension single-cell RNA sequencing has become high-throughput, it loses the spatial information encoded in the position of a cell from a tissue or organism.

## Challenge
* In order to evaluate methods that reconstruct the location of single cells in the Drosophila embryo using single-cell transcriptomic data, DREAM and SAGE Bioinformatics organized the DREAM Single-Cell Transcriptomics challenge.
* By providing public availability of RNA sequencing data, they devised a scoring and cross-validation scheme to evaluate the robustness and effectiveness of the top performing algorithms.

## Results
* The best algorithm known thus far was developed by Robert P. Zinzen et al. in a paper "The Drosophila Embryo at Single Cell Transcriptome Resolution" where they formulated an approach maximizing the Matthew's correlation coefficient between marker genes and the single cell sequences.
* The 34 participating teams used an array of methods and results show that the selection of genes was essential for accurately locating the cells in the embryo. This strategy led to the identification of archetypal gene expression patterns and spatial markers as participants were able to correctly localize rare subpopulations of cells, accurately mapping both spatially restricted and scattered groups of cells.

## Our approach
Our team used a combination of optimization strategies utilizing data imputation, optimization algorithms, and deep learning. For example, we tried using data from gene2function to improve our features. This is done with manifold leraning. See notebook in this folder.
