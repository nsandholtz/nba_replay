library(dplyr)
library(rstan)

# Load data and utils ----

source("./code/model_fitting_utils.R")
dat = read.csv("./data/2013_11_01_MIA_BRK_formatted.csv")

# Stan inputs ----
reward_stan_input = get_reward_inputs(dat = dat)

# Fit model ----

n_iter = 200
n_warmup = 50
n_chains = 2
options(mc.cores = 2)
begin_time <- proc.time()
reward_mod <- stan(file = "~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/stan_models/reward_model.stan",
                   data = reward_stan_input$stan_input,
                   iter = n_iter,
                   warmup = n_warmup,
                   chains = n_chains)
run_time <- proc.time() - begin_time
print(run_time) 

# Check diagnostics ----

# Means
mu_sum = summary(reward_mod, pars = "mu")$summary
psi_sum = summary(reward_mod, pars = "psi")$summary
varphi_sum = summary(reward_mod, pars = "varphi")$summary
xi_sum = summary(reward_mod, pars = "xi")$summary

# SDs
summary(reward_mod, pars = "sigma_mu")$summary
summary(reward_mod, pars = "sigma_psi")$summary
summary(reward_mod, pars = "sigma_varphi")$summary
summary(reward_mod, pars = "sigma_xi")$summary

# Saving output for simulation
mu_draws = extract(reward_mod, pars = "mu")$mu[1:300,]
dimnames(mu_draws)[[2]] = reward_stan_input$L1_dimnames
names(attr(mu_draws, "dimnames"))[2] = "state"
str(mu_draws)

xi_draws = extract(reward_mod, pars = "xi")$xi[1:300,]
dimnames(xi_draws)[[2]] = reward_stan_input$L3_dimnames
names(attr(xi_draws, "dimnames"))[2] = "region"
str(xi_draws)

saveRDS(mu_draws, "./model_output/mu_draws.rds")
saveRDS(xi_draws, "./model_output/xi_draws.rds")
