---
title: "Histogram Comparison 방법들"
date: 2020-08-22
categories: [Metric]
tags: [Machine Learning]
use_math: true
---

서로 다른 두 histogram *H1*과 *H2*의 차이(혹은 유사도)를 측정할때 주로 쓰이는 metric 4가지를 알아보고자 한다.

1. **Cross Correlation**
    
    $$
    d(H_1,H_2) = \frac{\sum_I (H_1(I) - \bar{H_1}) (H_2(I) - \bar{H_2})}{\sqrt{\sum_I(H_1(I) - \bar{H_1})^2 \sum_I(H_2(I) - \bar{H_2})^2}}
    $$
    
    where
    
    $$
    \bar{H_k} = \frac{1}{N} \sum _J H_k(J)
    $$
    
     이 식의 형태는 cross-covariance를 normalization한 식과 일치한다. 즉 sliding inner product를 통해서 두 histogram의 분포가 얼마나 유사한지를 나타내는 식이다. 값의 범위는 0~1이고 1에 가까울 수록 유사한 것이고 0에 가까울 수록 분포가 다른 것이다.
    
2. **Chi-Square**
    
    $$
    d(H_1,H_2) = \sum _I \frac{\left(H_1(I)-H_2(I)\right)^2}{H_1(I)}
    $$
    
     chi-square는 한국어로는 카이제곱이라고 부르는데 기준이 되는 histogram $H_1(I)$ 의 분포와 비교해서 이러한 분포가 실제로 일어날 법한 일인지에 대한 척도라고 하는데, 
    
    우리가 기록해서 보유하고 있는 histogram을 기준으로 현재 이러한 histogram 나타날 확률이 얼마나 되는지를 나타낸다고 보면 될 것 같다.
    
     
    
3. **Intersection**
    
    $$
    d(H_1,H_2) = \sum _I \min (H_1(I), H_2(I))
    $$
    
    이 방법은 가장 직관적으로 알기 쉬운 방법인데 말 그대로 두 histogram이 겹치는 넓이를 구하는 방법이다. 보통 histogram을 normalize해서 사용하는데 normalize한다는 가정 하에는 1에 가까울수록 유사하고 0에 가까울수록 분포가 다르다는 것을 의미한다.
    
4. **Bhattacharyya distance**
    
    $$
    d(H_1,H_2) = \sqrt{1 - \frac{1}{\sqrt{\bar{H_1} \bar{H_2} N^2}} \sum_I \sqrt{H_1(I) \cdot H_2(I)}}
    $$
    
     이 읽기 힘들게 생긴 이름을 가진 식은 한국어로는 바타차야 거리라고 읽는다고 한다. 대부분 이 식을 두고 bhattacharyya distance라고 하는데 실제로 식은 Helinger distance에 해당한다. 왜 이렇게 명명했는지는 잘 모르겠지만 helinger distance는 bhattacharyya distance와 term을 공유하는 변형된 형태지만 triangle inequality를 만족해서 이러한 형태를 쓰는 것으로 보인다.
    
     이 방법은 두 확률 분포의 distance를 유사도를 측정하는 방법이라고 하는데 histogram을 histogram의 전체 크기로 나눠서 합이 1로 만들어 확률분포 처럼 만들어서 histogram의 분포의 유사도를 구한다고 보면 된다.
    
    식을 통해서 의미를 살펴보면 두 histogram의 bin이 중첩되는 영역의 크기를 모두 더한 뒤 1에서 뺴주는 형태의 식으로 1에 가까울 수록 중첩되는 영역이 적은 것이고 0에 가까울 수록 중첩되는 영역이 커지는 것이다. 따라서 0~1의 값을 가지고 1에 가까울 수록 불일치에 가깝고 0에 가까울 수록 일치에 가까운 것을 의미한다.