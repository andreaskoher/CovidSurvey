# CovidSurvey

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://andreaskoher.github.io/CovidSurvey.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://andreaskoher.github.io/CovidSurvey.jl/dev) -->
[![Build Status](https://travis-ci.com/andreaskoher/CovidSurvey.jl.svg?branch=master)](https://travis-ci.com/andreaskoher/CovidSurvey.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/andreaskoher/CovidSurvey.jl?svg=true)](https://ci.appveyor.com/project/andreaskoher/CovidSurvey-jl)
[![Coverage](https://codecov.io/gh/andreaskoher/CovidSurvey.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/andreaskoher/CovidSurvey.jl)
[![Coverage](https://coveralls.io/repos/github/andreaskoher/CovidSurvey.jl/badge.svg?branch=master)](https://coveralls.io/github/andreaskoher/CovidSurvey.jl?branch=master)
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-red.svg)

# The Model:

### Model Priors
<p align="center"><img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/523fa2d015851efb1d567be0f4b5acf8.svg?invert_in_darkmode" align=middle width=447.6707103pt height=179.70650444999998pt/></p>

- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.649225749999989pt height=14.15524440000002pt/>: initial number of infected (seeding)
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/f50853d41be7d55874e952eb0d80c53e.svg?invert_in_darkmode" align=middle width=9.794543549999991pt height=22.831056599999986pt/>: dispersion parameter / "uncertainty" of observed hospitalizations
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/12d208b4b5de7762e00b1b8fb5c66641.svg?invert_in_darkmode" align=middle width=19.034022149999988pt height=22.465723500000017pt/>: initial reproduction number
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/>: probability to be hospitalized if infected
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/cbf94b0a9fb4d880661a1ab1549ca813.svg?invert_in_darkmode" align=middle width=11.638184249999991pt height=14.15524440000002pt/>: random walk
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/8cda31ed38c6d59d14ebefa440099572.svg?invert_in_darkmode" align=middle width=9.98290094999999pt height=14.15524440000002pt/>: typical step size of random walk

### Transmission Model

<p align="center"><img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/1490d26f597dee44a672917f75afddfd.svg?invert_in_darkmode" align=middle width=432.39299234999993pt height=154.97741324999998pt/></p>


- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/8b96a609d1c1c81c5ff51536677febdf.svg?invert_in_darkmode" align=middle width=12.19184174999999pt height=22.465723500000017pt/>: number of newly infected at time t
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/9f8bba50b95de09625626ddafa0698eb.svg?invert_in_darkmode" align=middle width=15.04571639999999pt height=22.465723500000017pt/>: fraction of the susceptible population
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/7f8a20dacaccab775d1e690bcf0f49e1.svg?invert_in_darkmode" align=middle width=17.447266649999992pt height=22.465723500000017pt/>: number of secondary infections / time-varying reproduction number. changes in <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/7f8a20dacaccab775d1e690bcf0f49e1.svg?invert_in_darkmode" align=middle width=17.447266649999992pt height=22.465723500000017pt/> are captured by the random walk term <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/cbf94b0a9fb4d880661a1ab1549ca813.svg?invert_in_darkmode" align=middle width=11.638184249999991pt height=14.15524440000002pt/>.
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/69304177ce432541b67b103783dfade3.svg?invert_in_darkmode" align=middle width=13.47643439999999pt height=14.15524440000002pt/>: distribution of time to secondary infections, also known as serial interval or next generation distribution


### Observation Model

<p align="center"><img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/3484260a363ca938f18d1eb8a0988c32.svg?invert_in_darkmode" align=middle width=454.21246694999996pt height=73.62578024999999pt/></p>


- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/39e731509d35a821a251fe62866ee4a4.svg?invert_in_darkmode" align=middle width=14.50919249999999pt height=22.465723500000017pt/>: observed hospilalization count
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/371fd45e7034625fe91e89b9280894a2.svg?invert_in_darkmode" align=middle width=13.02522374999999pt height=14.15524440000002pt/>: expected hospitalizations according to the model
- <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/617b82894383e2565c9cee47a8cf128b.svg?invert_in_darkmode" align=middle width=14.336104199999989pt height=14.15524440000002pt/>: time from infection to hospitalization, which is assumed to be a sum of two independent random times: the incubation period and time between onset of symptoms and hospitalization:

<p align="center"><img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/05e971654874c3e4442f000be86323cc.svg?invert_in_darkmode" align=middle width=314.29835249999996pt height=16.438356pt/></p>
where in this case the <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/e0e567f1ef3f1ab763500f631fd70a40.svg?invert_in_darkmode" align=middle width=56.73533909999999pt height=22.465723500000017pt/> is parameterized by its mean and coefficient of variation.

the model itself is a function that expects a set of parameters <img src="https://rawgit.com/andreaskoher/CovidSurvey (fetch/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> and returns the loglikelihood as well as the time-varying reproduction number and the expected daily hospitalization count.

All credits for the model go to Imperial College London:
[1] [Flaxman, S., Mishra, S., Gandy, A. et al. Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe. Nature 584, 257–261 (2020)](https://www.nature.com/articles/s41586-020-2405-7)
[2] [R-package Epidemia](https://imperialcollegelondon.github.io/epidemia/index.html)
