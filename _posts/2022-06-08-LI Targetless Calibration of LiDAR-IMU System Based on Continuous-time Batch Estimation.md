---
title: "Targetless Calibration of LiDAR-IMU System Based on Continuous-time Batch Estimation"
date: 2022-06-08 15:43:00 +0900
categories: [calibration]
tags: [calibration]
image: 
    src: /assets/img/licalib/LICALIBSURFEL.png
    width: 1000
    height: 400
---

# Targetless Calibration of LiDAR-IMU System Based on Continuous-time Batch Estimation

링크: [https://arxiv.org/pdf/2007.14759.pdf](https://arxiv.org/pdf/2007.14759.pdf)

![licalib-pipeline](/assets/img/licalib/LICALIBPIPELINE.png){: .align_center}
_LI-Calib의 전체 pipeline_

1. imu의 초기 orientation을 (0, 0, 1)로 두고 measurement 값을 이용해 b-spline curve를 얻고 lidar odometry는 ndt를 사용해 어느정도 pose의 방향 변화가 생길때까지 trajectory를 쌓은 후에 lidar로 부터 얻은 pose들의 orientation과 같은 timepoint에서 imu 측정값으로 부터 얻은 b-spline curve에서의 orientation 값을 이용해 둘 사이의 extrinsic rotation을 initialize 한다.
