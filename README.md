# Contrastive Viewpoint-aware Shape Learning for Long-term Person Re-Identification

This repository contains the code and resources for the publication titled "Contrastive Viewpoint-aware Shape Learning for Long-term Person Re-Identification". The publication proposes a method for person re-identification that leverages body shape for matching people across long durations and viewpoint variations.

## Overview

Traditional approaches for Person Re-identification (ReID) rely heavily on modeling the appearance of persons. This measure is unreliable over longer durations due to the possibility for changes in clothing or biometric information. Furthermore, viewpoint changes significantly degrade the matching ability of these methods. In this paper, we propose “Contrastive Viewpoint-aware Shape Learning for Long-term Person Re-Identification” (**CVSL**) to address these challenges. Our method robustly extracts local and global texture-invariant human body shape cues from 2D pose using the Relational Shape Embedding branch, which consists of a pose estimator and a shape encoder built on a Graph Attention Network. To enhance the discriminability of the shape and appearance of identities under viewpoint variations, we propose Contrastive Viewpoint-aware Losses (CVL). CVL leverages contrastive learning to simultaneously minimize the intra-class gap under different viewpoints and maximize the inter-class gap under the same viewpoint. Extensive experiments demonstrate that our proposed framework outperforms state-of-the-art methods on long-term person Re-ID benchmarks.

## Installation

See [INSTALL.md](docs/INSTALL.md).

## Features

Supported CNN backbones
c2dres50: C2DResNet50
i3dres50: I3DResNet50
ap3dres50: AP3DResNet50
nlres50: NLResNet50
ap3dnlres50: AP3DNLResNet50
