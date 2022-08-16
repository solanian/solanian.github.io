---
title: "Region Growing Segmentation 정리"
date: 2021-07-24
categories: [Segmentation]
tags: [Machine Learning, Pointcloud]
use_math: true
---

Reference: [https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html](https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html)

PCL에서 사용되는 Region Growing Segmentation의 방법론을 pcl 공식 문서를 참조하여 정리하였다. 

- Algorithm 개요
    - Inputs:
        - Point cloud=$\\{P\\}$
        - Point normals=$\\{N\\}$
        - Point curvatures=$\\{C\\}$
        - Neighbor finding function $\Omega(.)$
        - Curvature threshold $c_{th}$
        - Angle threshold $\theta_{th}$
    - Initialize:
        - Region list $R\leftarrow\emptyset$
        - Available points list $\\{A\\}\leftarrow\\{1,...,\|P\|\\}$
    - Algorithm:
        - While $\\{A\\}$ is not empty do
            - Current region $\\{R_c\\}\leftarrow\emptyset$
            - Current seeds $\\{S_c\\}\leftarrow\emptyset$
            - Point with minimum curvature in $\\{A\\}\rightarrow P_{min}$
            - $\\{S_c\\}\leftarrow\\{S_c\\}\cup P_{min}$
            - $\\{R_c\\}\leftarrow\\{R_c\\}\cup P_{min}$
            - $\\{A\\}\leftarrow\\{A\\}-P_{min}$
            - for $i=0$ to size $(\\{S_c\\})$ do
                - Find nearest neighbors of current seed point $\\{B_c\\}\leftarrow\Omega(S_c\\{i\\})$
                - for $j=0$ to size $(\\{B_c\\})$ do
                    - Current neighbor point $P_j\leftarrow B_c\\{j\\}$
                    - If $\\{A\\}$ contains $P_j$ and $cos^{-1} \( \| \( N \\{ S_c \\{ i \\} \\},N \\{ S_c \\{ j \\} \\} \) \| \) < \theta_{th}$ then
                        - $\\{R_c\\}\leftarrow\\{R_c\\}\cup P_j$
                        - $\\{A\\}\leftarrow\\{A\\}-P_j$
                        - If $c\\{P_j\\}<c_{th}$ then
                            - $\\{S_c\\}\leftarrow\\{S_c\\}\cup P_j$
                        - end if
                    - end if
                - end for
            - end for
            - Add current region to global segment list $\\{R\\}\leftarrow\\{R\\}\cup\\{R_c\\}$
        - end while
        - Return $\\{R\\}$

간단하게 정리하자면 minimum curvature를 가지는 point를 seed point로 삼아 nearest point들을 얻어서 normal의 cosine distance가 threshold 값 이하라면 같은 region에 포함시키는 방식으로 segmentation을 하는 알고리즘이다.