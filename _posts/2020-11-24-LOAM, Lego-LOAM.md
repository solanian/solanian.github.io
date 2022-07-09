---
title: "LOAM, Lego-LOAM"
date: 2020-11-24
categories: [SLAM]
tags: [SLAM]
image: 
    src: /assets/img/licalib/LICALIBSURFEL.png
    width: 1000
    height: 400
use_math: true
---


# LOAM, Lego-LOAM

Lidar Odometry에서 많이 쓰이는 LOAM 계열의 논문들을 리뷰해 보고자 한다.

# LOAM

paper: LOAM: Lidar Odometry And Mapping in Real-time (or Low-drift and Real-time Lidar Odometry and Mapping)

LOAM은 논문 제목대로 Lidar Odometry And Mapping의 줄임말로 high accuracy ranging과 inertial measurement 없이 Low-drfit, Low-computational complexity를 가진 odometry라고 한다.

Odometry는 아주 복잡한 문제인 SLAM 문제의 일부이며 이 odometry를 아래처럼 두 부분으로 나누어서 해결했다고 한다.

- motion estimation을 위해 높은 frequency로 돌아가지만 신뢰도가 비교적 낮은 odometry
- 위의 알고리즘에 비해 frequency가 낮지만 fine registration을 할 수 있는 mapping

LOAM의 전제 시스템은 다음과 같다.

![LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/637424184118100680.jpeg](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/637424184118100680.jpeg)

1. Lidar input이 들어오면 registration을 해서 Lidar odometry를 한다.
2. Odometry의 결과 pose를 이용해 Lidar mapping을 한다.
3. Mapping의 결과와 Odometry의 pose결과를 합쳐서 finagl pose를 얻는다.
    - Lidar Odometry
        - Feature Point Extraction
            
            Registration에 사용되는 feature는 edge와 plane이다. 그래서 point마다 edge point인지 planar point인지를 구별하기 위해서 smoothness를 구해서 이를 판별하였다.
            
            $$
            c=\frac{1}{|S||X^L_{(k, i)}|}||\sum_{j\in S ,\ j \neq i}(X^L_{(k, i)}-X^L_{(k, j)})||
            $$
            
            여기서 $S$는 표현이 애매한데 내가 이해한 바로는 point $i$와 consecutive한 point들의 set이며 양쪽 side를 반반씩 가지고 있는 set리고 한다. LeGO-LOAM 구현코드에서는 양쪽 point 5개씩 사용한것을 봐서는 결국 양쪽에 n개씩 사용한다는 뜻인것 같다. $\begin{vmatrix}S\end{vmatrix},\begin{vmatrix}X^L_{(k,i)}\end{vmatrix}$는 거리, set의 크기에 대한 normalize term역할을 한다. 
            
            그래서 이 maximum $c$들을 edge point 로 minimum c들을 planar point로 한다고 하는데 식을 보면 edge인 경우에 현재 point를 기준으로 consecutive한 point가 어느 한쪽으로 몰려있기 때문에 $c$의 값이 커지고 planar의 경우 평평할수록 현재 point를 지나는 대칭형태의 line이 되므로 값들이 상쇄되어 0에 가까워진다.
            
            그리고 feature point를 골고루 뽑기 위해 90도씩 4개의 subregion으로 나누고 각각의 region마다 threshold를 만족하는 edge는 최대 2개, planar는 최대 4개를 찾도록 하였다.
            
            ![LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled.png](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled.png)
            
            그리고 feature point를 골고루 뽑아야 하므로 주변에 feature point가 존재하면 뽑지 않도록 하였고 occluded region의 boudary또한 실제로는 planar인데 edge로 뽑힐 가능성도 있으므로 뽑지 않으며 laser beam에 parallel한 surface는 대체로 unreliable하므로 이또한 뽑지 않도록 한다. 이런 과정을 통해 edge point와 planar point는 아래와 같이 뽑힌다.
            
            ![노란색: edge 빨간색: planar](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%201.png)
            
            노란색: edge 빨간색: planar
            
        - Finding Feature Point Correspondence
            
            ![LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%202.png](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%202.png)
            
            Feature를 찾았으면 registration을 위해서 서로 다른 scene의 feature들 간에 correpondence를 생성해야한다. $P_{k+1}$에서 찾은 edge와 planar를 $E_{k+1}, H_{k+1}$이라 하면 이 point들에 대해 $P_k$에 있는 point중 nearest point를 찾아 correspondece를 만든다고 한다.
            
            edge point: $i\in E_{k+1}$에 대해서 $P_k$에서의 nearest point를 $j$라고 하고 edge line을 형성하기 위해서는 2개의 point가 필요하므로 $j$의 consecutive point $l\in P_k$를 구하고 이 $j, l$에 대해 smoothness를 계산해 둘 다 edge point라면 $(j,l)$이 이루는 edge line과 $i$ 사이에 correspondence를 만들고 다음과 같은 식으로 correspondence의 distance를 구한다.
            
            $$
            d_{\epsilon}=\frac{|(\tilde{X}^L_{(k+1,i)}-\bar{X}^L_{(k, j)})\times(\tilde{X}^L_{(k+1,i)}-\bar{X}^L_{(k,l)})|}{|\bar{X}^L_{(k, j)}-\bar{X}^L_{(k,l)}|}
            $$
            
            planar point: $i\in H_{k+1}$에 대해서 $P_k$에서의 nearest point를 $j$라고 하고 plane을 형성하기 위해서는 3개의 point가 필요하므로 세 점이 한 직선을 만들지 않도록 $j$의 nearest neighbor 2개의  $l,m\in P_k$를 구하고 이 $j,l,m$에 대해 smoothness를 계산해 셋 모두 planar point라면 $(j,l,m)$이 이루는 plane과 $i$사이에 correspondence를 만들고 다음과 같은 식으로 correspondence의 distance를 구한다.
            
            $$
            d_H=\frac{|(\tilde{X}^L_{(k+1,i)}-\bar{X}^L_{(k,j)})\{(\bar{X}^L_{(k,j)}-\bar{X}^L_{(k,l)})\times(\bar{X}^L_{(k,j)}-\bar{X}^L_{(k,m)})\}|}{|(\bar{X}^L_{(k,j)}-\bar{X}^L_{(k,l)})\times(\bar{X}^L_{(k,j)}-\bar{X}^L_{(k,m)})|}
            $$
            
        - Motion Estimation
            
            위에서 구한 correspondence를 이용해 두 frame사이의 motion estimation을 해야한다.  그런데 lidar point는 모든 point가 동시에 찍혀 나오는것이 아니라 일정한 속도로 sweep을 하면서 한 frame을 완성하는 것이기에 linear interpolation을 해줘야 한다. 그래서 transformation을 구한 시점 $t$에서의 6-DOF transformation을 $T^L_{k+1}=[t_x,t_y,t_z,\theta_x,\theta_y,\theta_z]^T$이라고 하고 transform 하는 point의 index를 $i$라고 하면 point $i$에 한 transformation은 다음과 같다.
            
            $$
            T^L_{(k+1,i)}=\frac{t_i-t_{k+1}}{t-t_{k+1}}T^L_{k+1}
            $$
            
            motion estimation을 위해서는 두 frame에서 찾은 feature들 사이의 transform matrix를 구해야한다. 즉 아래의 식을 만족하는 transformation을 구해야한다.
            
            $$
            X^L_{(k+1,i)}=R\tilde{X}^L_{(k+1,i)}+T^L_{(k+1,i)}(1:3)
            $$
            
            여기서 $R$은 optimization을 위해서 $T^L_{(k+1,i)}(4:6)$을 Rodrigues formula를 통해서 구한 $\omega$의 skew symmetric matrix이다. 
            
            위의 문제를 optimization problem으로 풀기 위해 correpondence의 거리가 가까워진다는 것은 제대로 registration(motion estimation)이 이뤄졌다는 의미이므로 아래와 같이 edge correspondence와 planar correspondence를 cost로 삼는다. 
            
            $$
            f_{E}(X^L_{(k+1,i)},T^L_{k+1})=d_E, i\in E_{k+1} \\ f_{H}(X^L_{(k+1,i)},T^L_{k+1})=d_H, i\in H_{k+1} \\ \rightarrow f(T^L_{k+1})=d
            $$
            
            optimization에 사용된 알고리즘은 Levenberg-Marquardt이다.
            
            $$
            T^L_{k+1}\leftarrow T^L_{k+1}-(J^TJ+\lambda diag(J^TJ))^{-1}J^Td
            $$
            
        - Lidar Odometry Algorithm
            
            ![LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%203.png](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%203.png)
            
            lidar odometry에 사용된 알고리즘을 정리하자면 위와 같다. 앞에 소개한 방법론들을 순차적으로 진행하여 얻은 transform을 통해 odometry를 한다.
            
- Lidar Mapping
    
    ![LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%204.png](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%204.png)
    
    lidar mapping은 odometry보다 더 낮은 빈도로 실행되고 sweep이 완성된 후에만 실행된다. odometry에서 구한 motion으로 untwist된 point cloud $$\bar{P}_{k+1}$$을 world coordinate상의 map에 registration하는 과정이다.
    
    $Q_k$를 sweep $k$가 끝난 시점에서의 pose주변의 cubic area내에 존재하는 map point들의 set이라 하고 $\bar{P}_{k+1}$을 mapping을 통해서 가장 최근에 얻은 transformation인 $T^W_k$으로 transform한 point를 $$\bar{Q}_{k+1}$$이라 하면 odometry에서 한 것처럼 feature extraction, finding correspondence, motion estimation을 통해서 $$\bar{Q}_{k+1}$$을 $$Q_k$$에 registration을 한다.
    
    mapping에서 $$\bar{Q}_{k+1}$$에 대한 feature extration은 이미 odomery에서 했기 때문에 그대로 사용한다. odometry가 10Hz로 돌아가고 mapping이 1Hz로 돌아가기 때문에 10배의 feature를 사용하게 된다. 
    
    correspondence 생성은 $\bar{Q}_{k+1}$의 각 feature point마다 주변에 존재하는 $Q_k$의 point의 set $S'$을 구하고 $S$에 대해서 matrix decomposition을 통해 eigenvalue와 eigenvector를 구하고 eigenvalue에서 dominant한 value의 갯수가 2개면 plane, 1개면 edge라고 판별하고 edge line과 planar patch의 position은 $S'$의 geometric center로 삼아서 feature point과 이 center사이의 corresopndence를 생성한다.
    
    optimization은 마찬가지로 Levenberg-Marquardt를 사용해서 transformation을 구한다.
    
    ![LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%205.png](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%205.png)
    
    그리고 마지막으로 pose integration은 mapping이 상대적으로 fine registration이기에 이미 존재하는 mapping에 odometry pose를 쌓는방식으로 한다. 따라서 실시간으로 얻는 pose의 결과는 $T^L_{k+1}T^W_K$가 되는 것이다.
    

# LeGO-LOAM

paper: LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry And Mapping on Variable Terrain

LeGO-LOAM은 ground plane의 존재를 이용해 lightweight한 real time 6DOF pose estimation을 했다고 한다.

noise filtering을 위해 segmentation을 해서 보다 robust한 결과를 얻었다고 했다.

computation expense를 줄였지만 성능은 LOAM과 비슷하거나 더 낫다고 했다.

- System Overview
    
    ![LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%206.png](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%206.png)
    
    LeGO LOAM의 전제 시스템은 다섯 부분으로 나뉜다.
    
    Segmentation 모듈에서 Point Cloud → Range Image → Segmented Point의 과정을 진행하고 Feature Extration 모듈에서 Segmented Cloud로 부터 Feature를 뽑고 이 feature를 이용해 odometry와 mapping을 하고 두 pose를 integration한다고 한다.
    

![LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%207.png](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%207.png)

- Segmentation
    
    Point Cloud (그림 a)를 $1800\times16$ 크기의 range image로 변환하고  range image에서의 pixel value는 point의 sensor로 부터의 euclidean distance로 한다. 그리고 Segmentation 이전에 ground extration을 하고서 segmentation을 한다고 하는데 이 ground extration은 column-wise evaluation방법을 통해서 한다고 하는데 이 방법은 range image 상에서 column-wise slope를 이용해서 threshold 미만이면 ground로 판단해서 ground point들을 뽑아낸다고 했다.  이렇게 ground point들의 index를 제외한 나머지 range image에서 image-based segmentation을 한다. 여기서 robustness를 위해 point 갯수가 30개 미만인 segment들은 사용하지 않는다. 이 결과로 segmented point과 groud point (그림 b)를 얻는다.
    
- Feature Extraction
    
    앞에서 추출한 segmented point와 ground point에서 feature를 뽑는 과정이다.  LOAM에서와 같은 smoothness를 정의해서 사용한다.
    
    $$
    c=\frac{1}{|S|||r_i||}||\sum_{j\in S,j\neq i} (r_j-r_i)||
    $$
    
    $S$는 range image상에서 같은 row에 있는 연속적인 point를 사용하였고 LeGO-LOAM 구현 코드에서는 앞뒤로 5개의 point를 사용하였다. 그리고 이 smoothness를 가지고 edge와 planar를 구분한다. (자세한것은 LOAM에서의 설명 참조)
    
    다른 점은 edge point로 판별되었지만 ground point일 경우는 feature로 사용하지 않으며 $60^{\circ}$씩 6개의 sub-image로 나눠서 edge 와 planar point들을 뽑는다. $\mathbb{F}_e,\mathbb{F}_p$는 6개의 sub image에 있는 모든 feature들의 set이며 $$F_e, F_p$$는 각각의 sub image에 존재하는 feature들의 set이며 $$n_{F_e}, n_{F_p}, n_{\mathbb{F}_e},n_{\mathbb{F}_e}$$는 각각 2, 4, 40, 80으로 정했다. feature extracion을 통해 얻은 feature들은 위의 그림 c와 d에 나타나 있다.
    

![LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%208.png](/assets/img/LOAM,%20Lego-LOAM%20e6abee5cd52d4cfa929a3dd59f13cf36/Untitled%208.png)

- Lidar Odometry
    
    Lidar odometry 모듈에서는 두 개의 연속된 scan 사이의 transformation을 feature들간의 correspondence를 이용해서 구한다. 이를 위해서는 $F^t_e,F^t_p$와 $\mathbb{F}^{t-1}_e,\mathbb{F}^{t-1}_p$사이에서 correspondence를 구하고 이를 optimizaion 해야한다.
    
    - Label Matching
        
        matching의 효율성을 위해서 모든 feature를 match에 이용하는것이 아니라 segmented point에서는 $F^t_e$와 $\mathbb{F}^{t-1}_e$ 사이의 correspondence만 찾고 ground point에서는 $F^t_p$와 $\mathbb{F}^{t-1}_p$사이의 correspondence만 LOAM에서와 같은 방식으로 찾는다.
        
    - Two-step LM Optimization
        
        optimization에서도 속도의 효율성을 높이기 위해서 6-DOF의 transform $[t_x,t_y,t_z,\theta_{roll},\theta_{pitch},\theta_{yaw}]^T$를 한번에 optimization하는 것이 아니라 두 개의 단계로 나눠서 optimization을 진행한다.
        
        1. $[t_Z,\theta_{roll},\theta_{pitch}]$를 ground plane을 이용하여 즉 $F^t_p$와 $\mathbb{F}^{t-1}_p$사이의 correspondence의 distance를 줄이는 방향으로 optimization 한다.
        2. $[t_x,t_y,\theta_{yaw}]$를 $F_e^t$,$\mathbb{F}^{t-1}_e$ 사이의 distance를 줄이는 방향으로 optimization을 한다.
        
        그리고 각각의 과정에서 optimize하는 parameter외의 나머지 parameter는 constraint로 삼아서 optimize한다. 
        
        이렇게 하는 이유는 ground plane만 사용해도 $t_z,\theta_{roll},\theta_{pitch}$를 optimize할 수 있으므로 parameter를 3개씩 나눠서 optimization을 진행해도 optimization이 가능하며 parameter수를 줄이면 LM알고리즘의 특성상 search space가 작아지기 때문에 6개의 한번에 optimization하는것보다 3개씩 두개의 과정으로 나누어 optimization하는것이 더 빠르기 때문이다. 이 방법이 accruracy를 높이는데에 도움이 되었을 뿐만 아니라 실제로 35%정도의 computation time이 줄었다고 한다.
        
- Lidar Mapping
    
    Lidar Mapping 모듈은 낮은 빈도로 돌아가지만 pose transformation을 refine하기 위해 $\mathbb{F}^t_e, \mathbb{F}^t_p$의 feature들을 주변의 point cloud map $\bar{Q}^{t-1}$과 matching하고 L-M 알고리즘을 사용하여 transformaion을 구하는 모듈이다.
    
    LeGO-LOAM에서는 LOAM과는 달리 point cloud map을 저장할때 feature set $\{\mathbb{F}^t_e,\mathbb{F}^t_p\}$도 같이 저장한다. 여기서 $M^{t}=\{\{\mathbb{F}^1_e,\mathbb{F}^1_p\},\cdots,\{\mathbb{F}^t_e,\mathbb{F}^t_p\}\}$라고 하면 각각의 $M^t$에 대응하는 pose를 연결짓는 식으로 저장을 한다. 이 $M^{t-1}$로 부터 $\bar{Q}^{t-1}$을 얻는 방법은 두 가지가 있다.
    
    1. 현재 pose를 기준으로 저장된 feature들 중에서 주변 100m 이내에 있는 모든 pose들의 feature들을 불러오고 이 모든 feature들을 각 pose로 transform하고 합쳐서 surrounding map $\bar{Q}^{t-1}$을 얻는다.
    2. LeGO-LOAM을 pose-graph SLAM하고 통합해서 사용한다고 하면 sensor의 pose는 graph의 node로, feature set $\{\mathbb{F}^t_e,\mathbb{F}^t_p\}$은 각 node의 measurement로 모델링 할 수 있다. 그리고 lidar mapping의 pose estimation drift가 매우 작으므로 단기적으로는 pose의 drift가 없다는 가정하에 최근의 $k$개의 pose 즉 $\{\{\mathbb{F}^{t-k}_e,\mathbb{F}^{t-k}_p\},\cdots,\{\mathbb{F}^{t-1}_e,\mathbb{F}^{t-1}_p\}\}$을 사용해 $\bar{Q}^{t-1}$을 만든다.
    
    그리고 odometry와 같은 방식으로 correspondence를 생성한 후에 L-M optimization을 해서 transform을 얻는다. 여기에 추가적으로 loop closure detection을 해서 ICP와 같은 registration을 통해 추가적인 contraint를 얻으면 drift를 줄일 수 있을 것이라 한다.