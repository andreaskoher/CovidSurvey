# CovidSurvey

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://andreaskoher.github.io/CovidSurvey.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://andreaskoher.github.io/CovidSurvey.jl/dev) -->
[![Build Status](https://travis-ci.com/andreaskoher/CovidSurvey.jl.svg?branch=master)](https://travis-ci.com/andreaskoher/CovidSurvey.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/andreaskoher/CovidSurvey.jl?svg=true)](https://ci.appveyor.com/project/andreaskoher/CovidSurvey-jl)
[![Coverage](https://codecov.io/gh/andreaskoher/CovidSurvey.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/andreaskoher/CovidSurvey.jl)
[![Coverage](https://coveralls.io/repos/github/andreaskoher/CovidSurvey.jl/badge.svg?branch=master)](https://coveralls.io/github/andreaskoher/CovidSurvey.jl?branch=master)

# The Model:

### Model Priors
asdfadsf
\begin{align}
  \tau & \sim \mathrm{Exponential}(1 / 0.03) \\
  y & \sim \mathrm{Exponential}^{[0,1000]}(\tau)\\
  \phi  & \sim \mathcal{N}^{ + }(25, 10) \\
  R_0 & \sim \mathcal{N}^{[2,5]}(3.6, 0.8)\\
  \alpha &\sim \mathcal{N}^{[0,0.05]}(0.01, 0.01)\\
  \sigma &\sim \mathcal{N}^{[0,0.5]}(0.1,0.3)\\
  \epsilon_t &\sim \mathcal{N}(\epsilon_{t-1}, \sigma)
\end{align}

- $y$: initial number of infected (seeding)
- $\phi$: dispersion parameter / "uncertainty" of observed hospitalizations
- $R_0$: initial reproduction number
- $\alpha$: probability to be hospitalized if infected
- $\epsilon_t$: random walk
- $\sigma$: typical step size of random walk

### Transmission Model

\begin{align}
  I_{t} &= R_{t} S_{t}  \sum_{\tau = 1}^{t - 1} I_{\tau} \gamma_{t - \tau} \\
  S_{t} &= 1 - \sum_{\tau = 1}^{t - 1} \frac{I_{\tau}}{N}\\
  R_{t} &= \exp( \log(R_0) + \epsilon_t )\\
\end{align}


- $I_t$: number of newly infected at time t
- $S_t$: fraction of the susceptible population
- $R_t$: number of secondary infections / time-varying reproduction number. changes in $R_t$ are captured by the random walk term $\epsilon_t$.
- $\gamma_t$: distribution of time to secondary infections, also known as serial interval or next generation distribution


### Observation Model

\begin{align}
Y_{t} &\sim \mathrm{NegativeBinomial}(y_{t}, \phi)\\
y_{t} &= \alpha \sum_{\tau=1}^{t - 1} I_{\tau} \pi_{t - \tau}
\end{align}


- $Y_t$: observed hospilalization count
- $y_t$: expected hospitalizations according to the model
- $\pi_t$: time from infection to hospitalization, which is assumed to be a sum of two independent random times: the incubation period and time between onset of symptoms and hospitalization:

$$
\begin{equation*}
\pi \sim \mathrm{Gamma}(5.1, 0.86) + \mathrm{Gamma}(6.49, 0.92)
\end{equation*}
$$
where in this case the $\mathrm{Gamma}$ is parameterized by its mean and coefficient of variation.

the model itself is a function that expects a set of parameters $\theta$ and returns the loglikelihood as well as the time-varying reproduction number and the expected daily hospitalization count.

All credits for the model go to Imperial College London:
[1] [Flaxman, S., Mishra, S., Gandy, A. et al. Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe. Nature 584, 257–261 (2020)](https://www.nature.com/articles/s41586-020-2405-7)
[2] [R-package Epidemia](https://imperialcollegelondon.github.io/epidemia/index.html)
