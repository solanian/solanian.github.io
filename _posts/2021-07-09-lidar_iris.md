---
title: "LiDAR IRIS 정리"
date: 2021-07-09
categories: [SLAM]
tags: [SLAM]
use_math: true
---

paper: [LiDAR Iris for Loop-Closure Detection](https://arxiv.org/pdf/1912.03825.pdf)

![Untitled](../../assets/img/LiDAR%20IRIS%206be479f97b9b40eb8dc266783376cda1/Untitled.png)

![Untitled](../../assets/img/LiDAR%20IRIS%206be479f97b9b40eb8dc266783376cda1/Untitled%201.png)

LiDAR Iris는 빠르고 정확한 loop-closure detection을 위해 LiDAR-Iris image에 LoG-Gabor filter와 thresholding을 적용해서 binary signature image를 feature로 사용한다. 두 이미지의 hamming distance를 이용해 두 point cloud사이의 similarity를 fourier transform을 이용해 descriptor level에서  pose invariant하게 구할 수 있다고 한다.

기존의 global descriptor와 local descriptor는 각각 pose invariance문제와 descriptive power가 약하다는 문제가 존재했으며 learning-based의 경우 training data가 매우 많아야하고 다른 조건에서 취득된 데이터나 topology가 다양한 데이터에 대해서 generalize가 잘 되지 않는다는 문제가 있다고 해서 이를 해결하려고 한다고 한다.

이 방법은 Scan Context와 유사한 방식이지만 크게 세 가지 차이점이 있다고 한다.

- 주변의 높이 정보를 LiDAR-Iris의 pixel intensity로 삼았다.
- Loop-closure detection을 위해 LiDAR-Iris image로부터 discriminative한 binary feature map을 얻었다고 한다.
- Scan Context처럼 brute force matching을 할 필요 없이 LiDAR의 pose에 invariant한 loop-closure detection을 한다고 한다.

# Lidar Iris

## Generation of LiDAR-Iris Image Representation

![Untitled](../../assets/img/LiDAR%20IRIS%206be479f97b9b40eb8dc266783376cda1/Untitled%202.png)

Bird eye view로 point cloud를 projection하고 range와 yaw 값으로 discretization을 한다. pointcloud를 완벽하게 표현하기 위해서는 각각의 bin에 대해서 height, range, reflection, ring과 같은 정보를 담도록 해야한다. 이를 간단하게 하기 위해 같은 bin에 들어가는 모든 point의 정보를 8 bit로 encoding했다고 한다. 우선 scan context와 마찬가지로 angle, range를 기준으로 영역을 나누고 각 영역마다 값을 부여한다. 각 영역의 값을 구하기 위해 point의 최고, 최저 높이인 $y_h,y_l$을 센서마다 적당히 정해서 그 사이를 8등분 하고 각 영역마다 또 binary로 값을 부여할 수 있게 하는데 이를 $y_k, k\in[1, 8]$이라 한다. 그리고 $y_k$에 해당하는 영역 내에 point가 있다면 1, 없다면 0을 부여하여 binning을 하여 구한 8 bit의 이진수가 해당 $(r,\theta)$ 영역의 값이 된다. 

![Untitled](../../assets/img/LiDAR%20IRIS%206be479f97b9b40eb8dc266783376cda1/Untitled%203.png)

Iris recognition의 영향을 받아서 위의 그림과 같이 이 이미지를 strip형태로 폈다고 한다.

## Fourier transfom for a translation-invariant LiDAR Iris

![Untitled](../../assets/img/LiDAR%20IRIS%206be479f97b9b40eb8dc266783376cda1/Untitled%204.png)

Translation variation이 Lidar Iris Image를 matching할때 성능을 낮추는 요인중 하나가 될 수 있다. 이를 해결하기 위해 두 image사이의 translation을 estimate하기 위해서 Fourier Transform을 사용하였다. Fourier-based scheme에서는 rotation, scaling, translation을 모두 estimate할 수 있다. Frequency domain 상에서 rotation은 horizontal traslation, translation은 vertical translation으로 이어지면서 image pixel의 intensity에 변화를 준다. 이 점을 활용하여 FT를 한 후의 cross power spectrum의 inverse FT가 non-zero가 되는 frequency domain image상의 shift$(\delta_x, \delta_y)$ 를 구해 translation과 rotation을 구한다. 식은 다음과 같다.

$$
\hat{I_1}(w_x,w_y)\dot{e}^{i(w_x\delta_x+w_y\delta_y)}=\hat{I_2}(w_x,w_y)
$$

$$
\hat{Corr}=\frac{\hat{I_2}(w_x,w_y)}{\hat{I_1}(w_x,w_y)}=\frac{\hat{I_2}(w_x,w_y)\hat{I_1}(w_x,w_y)*}{|\hat{I_2}(w_x,w_y)\hat{I_1}(w_x,w_y)*|}=e^{-i(w_x\delta_x+w_y\delta_y)}
$$

$$
Corr(x,y)=F^{-1}(\hat{Corr})=\delta(x-\delta_x,y-\delta_y)
$$

$$
(\delta_x,\delta_y)=argmax_{x,y}\{Corr(x,y)\}
$$

## Binary feature extraction with LoG-Gabor filters

![Untitled](../../assets/img/LiDAR%20IRIS%206be479f97b9b40eb8dc266783376cda1/Untitled%205.png)

Represenation ability를 높이기 위해 LoG-Gabor filter를 사용해 Lidar Iris image로부터 추가적인 feature를 뽑아낸다. LoG-Gabor filter는 다양한 resolution으로 LiDAR IRIS 영역안에 data를 나타낼 수 있어서 fourier transform에 비해 같은 resolution과 position을 가지는 feature를 매칭하기 유리하다. Fourier transform의 frequency data는 매우 국소적(delta 함수)이기 때문이다. 여기서 1D LoG-Gabor filter를 사용하였다.

$$
G(f)=exp(\frac{-(\log(f/f_0))^2}{2(log(\sigma/f_0))^2})
$$

$f_0$는 center frequency, $\sigma$는 bandwidth로 filter의 parameter이다. Iris image의 각 row마다 1D LoG-Gabor filter를 적용하였으며 사용한 fillter는 4개이다. 여러가지 갯수의 filter를 사용해봤으나 4개일때가 결과가 제일 좋았다고 한다.

![Untitled](../../assets/img/LiDAR%20IRIS%206be479f97b9b40eb8dc266783376cda1/Untitled%206.png)

## Loop-Closure Detection with Lidar IRIS

Loop closure detection 과정에서는 log gabor filter를 이용해 구한 binary feature map 사이의 hamming distance를 구해 이 distance의 threshold 이하의 frame을 loop로 삼는다.

## 총평

극 좌표계를 사용해서 binning을 하면 어떤 식으로 translation error를 해결을 하나 싶었는데 이 논문에서는 fourier transform을 사용해서 해결하고자 한 것 같은데 여전히 frequency domain상에서의 vertical translation이 실제 translation 차이를 반영하는데 충분한 가에 대해서는 의문이 든다. 코드를 쭉 봤을 때는 보통 loop closure detection이 도는데 시간이 오래 걸려서 2-stage로 돌릴텐데 여기서는 그 부분에 대한 고려가 없이 전체 frame에 대해서 IRIS를 돌리는 것 같아 보였다. 실제 사용하려면 scan context에서 ring key로 candidate를 뽑는 부분 뒤에 scan matching을 하는 부분을 대체해서 써야 할 것 같다.