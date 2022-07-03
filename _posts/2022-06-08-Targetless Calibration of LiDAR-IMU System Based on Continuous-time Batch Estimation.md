---
title: "Targetless Calibration of LiDAR-IMU System Based on Continuous-time Batch Estimation"
date: 2022-06-08 15:43:00 +0900
categories: [calibration]
tags: [calibration]
image: 
    src: /assets/img/licalib/LICALIBSURFEL.png
    width: 1000
    height: 400
use_math: true
---

# Targetless Calibration of LiDAR-IMU System Based on Continuous-time Batch Estimation

링크: [https://arxiv.org/pdf/2007.14759.pdf](https://arxiv.org/pdf/2007.14759.pdf)

![licalib-pipeline](/assets/img/licalib/LICALIBPIPELINE.png){: .align_center}
_LI-Calib의 전체 pipeline_

## 요약

이 논문의 알고리즘을 간단하게 요약하자면 IMU로 부터 얻은 measurement 값과 중간 과정에서 구하는 extrinsic 값을 이용해 lidar sensor의 distortion을 보정해서 map을 생성하고 생성된 map에서 얻은 plane들을 구해 lidar frame에서의 point들이 각각의 point가 속한 map 상의 plane과의 거리를 cost로 삼아 더 나은 extrinsic 값을 얻고 이를 반복하면서 extrinsic 값을 계속 개선해 나가는 것이다.

## Methods

전체 pipeline이 돌아가는 과정을 살펴보면 다음과 같다.

### Initialization

Initialization 과정은 처음에는 extrinsic 값에 대한 정보가 전혀 없기 때문에 랜덤한 초기값을 줘서 lidar distortion을 보정하면 map 자체가 만들어질 수 없기 때문에 초기에 어느정도 사용 할만한 extrinsic 값을 구하는 과정이다.

1. IMU의 초기 orientation을 (0, 0, 1)로 두고 IMU measurement 값을 이용해 b-spline curve를 얻는다.

2. Lidar에서는 ndt로 odometry를 돌려서 어느정도 pose의 변화가 생길때 까지의(코드 상에서 odometry로 구한 pose들의 covariance의 minimum eigen value가 0.25이상이 될 때 까지) trajectory를 구한다.

3. 위에서 구한 두 pose trajectory를 이용해 lidar로 부터 얻은 pose들의 orientation과 같은 timepoint에서의 imu로 부터 얻은 orientation을 b-spline curve를 통해 일종의 interpolation 된 값을 가져와서 둘 사이의 extrinsic rotation을 initialize 한다. 자세한 내용은 아래와 같다.

    $$
    ^{I_k}_{I_{k+1}}q\otimes^{I}_{L}q=^{I}_{L}q\otimes^{L_k}_{L_{k+1}}q \\
    $$

    위의 식을 기반으로 하여 각각의 measurement마다 $$\begin{pmatrix}\begin{bmatrix}^{I_k}_{I_{k+1}}q\end{bmatrix}_L-\begin{bmatrix}^{L_k}_{L_{k+1}}q\end{bmatrix}_R\end{pmatrix}^I_Lq=0$$ 이라는 cost를 정의하고 outlier를 처리하기 위해 각각의 cost마다 weight을 ${\alpha_k}$로 heuristic하게 줘서 전체 식을 minimize하는 방식으로 optimization을 한다.

이 과정을 통해서 inertial to lidar extrinsic 값중에 rotation 값들만 얻게 된다.

### Data Association

![licalib-surfelmap](/assets/img/licalib/LICALIBSURFEL.png){: .align_center}
_Map에서 구한 plane들_

이 과정은 Initialization 과정을 통해서 구한 extrinsic값과 IMU measurement로 구한 b-spline을 이용해서 lidar scan의 rotation으로 인한 distortion을 보정하고 frame들의 point를 쌓아서 map을 만든다. 그 후에 map에서 plane에 가까운 point들을 추출하고 map에서의 각각의 plane마다 plane에 속한 point가 lidar frame에서의 어떤 point와 일치하는지를 association하여 optimization 과정 중에 바로 사용할 수 있도록 준비하는 과정이다. 코드를 봤을때 plane추출은 전체 scene을 grid로 나누고 각 grid마다 RANSAC을 사용해 plane을 구하고 각 plane마다 plane-likeness를 구하여 plane이 아닌 부분을 걸러냈다. 그 결과는 위의 그림과 같다.

이 과정을 통해 각 plane의 point들이 lidar frame상에서 어떤 point와 같은 point인지 association 되었는지를 구할 수 있다.

### Batch Optimization

Extrinsic 값을 refine하기 위해서는 optimization을 계속 해줘야하는데 optimization에 사용되는 cost function은 크게 세부분으로 나뉜다. angular velocity의 residual($r_{\omega}^{k}$), acceleration의 residual($r_{a}^k$), point-to-plane distance의 residual($r_{\mathcal{L}}^{j}$)이다. angular velocity와 acceleartion의 경우 단순하게 모든 timepoint에서 IMU의 measurement와 lidar의 measurement를 extrinsic 값을 이용해 한쪽을 transform하고 그 둘의 차이를 minimize하는 것이다. point-to-plane distance는 map frame상에서 plane과 plane에 속한 point들의 lidar frame상에서의 위치에서의 거리를 minimize 하는 것이다. 식으로 나타내면 아래와 같다.

$$ 
\hat{x}=argmin\begin{Bmatrix}\sum_{k\in\mathcal{A}}||r_a^k||^2_{\Sigma_{a}} + \sum_{k\in\mathcal{W}}||r_\omega^k||^2_{\Sigma_{\omega}} + \sum_{j\in\mathcal{L}}||r_\mathcal{L}^j||^2_{\Sigma_{L}} \end{Bmatrix} \\
r_a^k=^{I_k}a_m-^Ia(t_k)-b_a \\
r_\omega^k=^{I_k}\omega_m-^I\omega(t_k)-b_g \\
^{L_0}p_i=^I_L{R^T}\ \ ^{I_0}_{I_j}R\ \ ^I_LR\ \ ^{L_j}p_i+^{L_0}p_{L_j} \\
^{L_0}p_{L_j}=^I_L{R^T}\ \ ^{I_0}_{I_j}R \ \ ^Ip_L + ^I_L{R^T}\ \ ^{I_0}p_{I_j} - ^I_L{R^T}\ \ ^Ip_L \\
r^j_\mathcal{L}=\begin{bmatrix}{^{L_0}p_i^T} & {1}\end{bmatrix}\pi_j
$$

이전에 구한 extrinsic 값과 association값을 이용해 optimization과정을 거치면 더 나은 extrinsic 값을 얻을 수 있게 된다.

### Refinement

Refinement과정은 앞에서 나왔던 과정들을 그저 반복한다는 것에 대한 설명인데 결국은 cost들이 extrinsic 값에 dependent하기 때문에 더 나은 extrinsic 값을 얻게 된다면 angular velocity, acceleration residual은 수식에서 자명하게 줄어들 것이다. 그리고 point-to-plane distance residual의 경우 더 나아진 extrinsic 값을 이용해 lidar scan을 undistortion을 하고 이 point들을 쌓아서 mapping을 하게 되면 원래 plane인 point들에 대해서 더 명확하게 plane에 가깝게 mapping될 것이며 이 plane과 실제 point와의 거리는 lidar scan의 noise가 없다면 0이 되어야 한다. 이런식으로 data association -> optimization -> data association ->optimization...을 반복하면서 extrinsic 값을 개선해 나가는 과정이다.

## 총평

실제 구현해서 돌린 체감상 ndt를 point를 하나도 안 버리고 처음부터 끝까지 쌓으니까 상당히 오래걸리며 mapping을 하고 그 값을 그대로 다 들고 있으니 메모리도 엄청 잡아먹는다.

[TODO] 실제 실험결과 추가