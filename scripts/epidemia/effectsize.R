#!/usr/bin/env Rscript

library(dplyr)
library(optparse)
library(epidemia)
library(rstanarm)
library(ggplot2)
options(mc.cores = parallel::detectCores())
data("EuropeCovid")

#option_list <- list(
#  make_option(c("--region"),action="store", default="Denmark",help="Which region to run this for [default \"%default\"
#]"))
#opt <- parse_args(OptionParser(option_list=option_list))
#print(opt$region)

data <- read.csv(file = '/home/ankoh/dev/CovidSurvey/data/epidemia/denmark_predictors=CS-MG-MA.csv') %>% group_by(region)
data$hospits[1:6] <- NA
data$deaths[1:6] <- NA
data$cases[1:6] <- NA
head(data)

dates <- summarise(data, start = min(date), end = max(date))
dates

inf <- epiinf(gen = EuropeCovid$si, seed_days=6L, pop_adjust = FALSE)


hospit <- epiobs(formula = deahts ~ 1, i2o = EuropeCovid2$inf2death,
                 prior_intercept = normal(0,0.2), link = scaled_logit(0.02))

rt <- epirt(
  R(region, date) ~ 1 + rw(time = rw, prior_scale=0.02),
  link = scaled_logit(7), prior_intercept = normal(-1,1)
)

args <- list(rt = rt, inf = inf, obs = deaths, data = data, seed = 12345, refresh = 0)

pr_args <- c(args, list(algorithm = "sampling", iter = 2e2, chains = 4, control = list(max_treedepth = 12)))

fm <- do.call(epim, pr_args)

p <- plot_obs(fm, type = "deaths", levels = c(50, 95))
#fname <- sprintf("/home/ankoh/dev/CovidSurvey/reports/stanfit/2021-10-05/plot_deaths_region=%s.png", opt$region)
#ggsave(fname, p)

p <- plot_rt(fm, step = T, levels = c(50,95))
#fname <- sprintf("/home/ankoh/dev/CovidSurvey/reports/stanfit/2021-10-05/plot_rt_region=%s.png", opt$region)
#ggsave(fname, p)

sampled_rt <- posterior_rt(fm)
#fname <- sprintf("/home/ankoh/dev/CovidSurvey/reports/stanfit/2021-10-05/sampled_rt_region=%s.rds", opt$region)
#saveRDS(sampled_rt, fname)

sampled_deaths <- posterior_predict(fm)
#fname <- sprintf("/home/ankoh/dev/CovidSurvey/reports/stanfit/2021-10-05/sampled_deaths_region=%s.rds", opt$region)
#saveRDS(sampled_deaths, fname)
