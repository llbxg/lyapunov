# LYAPUNOV
リアプノフスペクトラムを求める必要があったので、作成しました。   

## Functions

連立常微分方程式を以下のように記述   
<img src="https://render.githubusercontent.com/render/math?math=$\frac{d X_j}{dt}=F_j(X_1,X_2,..,X_j)$">


### lyapunov_exponent_1
Perturbation Vector<img src="https://render.githubusercontent.com/render/math?math=${\mathbf w(t)}$">を直接求める。   

<img src="https://render.githubusercontent.com/render/math?math=$\frac{dw_j}{dt}=\sum_{k=1}^{j}(\frac{\partial F_j}{\partial X_k})_{\tilde \mathbf x} w_k$">

方向を考慮していないので、最大リアプノフ指数のみ求めている。   

### lyapunov_exponent_123
(図はそのうち追加します。)   
n次元の総空間の中に直交するmこの初期ベクトルの組を用意します。   
これらをm個の近接軌道に従って発展させていきながら、Lyapunov spectrumを求めます。   

<img src="https://render.githubusercontent.com/render/math?math=$\mathbf d_{0\bot}^{1}(\tau)= \mathbf d_0^1(\tau)$">


<img src="https://render.githubusercontent.com/render/math?math=$\mathbf d_{\bot}^k=\mathbf d^k-a_{k,1}\mathbf d_{\bot}^1-a_{k,2}\mathbf d_{\bot}^2-a_{k,3}\mathbf d_{\bot}^3-\dotsb-a_{k,k-1}\mathbf d_{\bot}^{k-1}$">
<img src="https://render.githubusercontent.com/render/math?math=$a_{k,i}=\frac{(\mathbf d^k,\mathbf d_{\bot}^i)}{{\lVert \mathbf d_{\bot}^i \rVert}^2}$">
<img src="https://render.githubusercontent.com/render/math?math=$\mathbf d_1^i(0)=\frac{\lVert\mathbf d_1^i(0)\rVert}{\lVert d_{1\bot}^i(\tau)\rVert}d_{1\bot}^i( \tau)$">
<img src="https://render.githubusercontent.com/render/math?math=$\sum_i^m{\lambda_i}=\lim_{n \to \infin}^{n-1}\frac{1}{n\tau}ln\frac{\lVert\mathbf d_k^1(\tau)\land d_k^2(\tau)\land d_k^3(\tau)\land\dotsb \land d_k^m(\tau)\rVert}{\lVert\mathbf d_k^1(0\land d_k^2(0)\land d_k^3(0)\land\dotsb \land d_k^m(0)\rVert }$">

## Test

> A Numerical Approach to Ergodic Problem of Dissipative Dynamical Systems   
> Ippei Shimada, Tomomasa Nagashima / Published: 01 June *1979*

こちらを参考に<img src="https://render.githubusercontent.com/render/math?math=$\sigma = 16.0, b=4.0, r=40.0$">の条件下で、   
Lorenz方程式を解きLyapunov spectrumの値を確認しました。   


## code
方程式を定義して、それを計算していきます。
サンプルとしてLorenzを載せておきます。   


### 方程式の定義
```Python
import numpy as np

from lib.ode import Ode

class Lorenz(Ode):
    def __init__(self, params, size):
        super().__init__(params, size)

    def _eq(self, u):
        r, s, b = self.params
        x, y, z = u

        x_dot=s*(y-x)
        y_dot=r*x-y-x*z
        z_dot=x*y-b*z

        return np.array([x_dot, y_dot, z_dot])

    def _jm(self, u):
        r, s, b = self.params
        x_1, x_2, x_3 = u

        jm = np.array([
            [   -s,   s,    0],
            [r-x_3,  -1, -x_1],
            [  x_2, x_1,   -b]
            ])

        return jm
 ```
 
### 計算
 
```Python
from lib.handle import Handle

from lorenz import Lorenz
from test import Test

l = Lorenz(params=(40, 16, 4), size=3)
h = Handle(l)
l_123 = h.lyapunov_exponent_123()
print(l_123)
h.graph()
 ```
