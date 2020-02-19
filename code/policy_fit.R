library(dplyr)
library(rstan)

# Load data and utils ----

source("./code/model_fitting_utils.R")
dat = read.csv("./data/2013_11_01_MIA_BRK_formatted.csv")

# Stan inputs ----
policy_stan_input = get_policy_inputs(dat = dat)

# Fit models ----

n_iter = 500
n_warmup = 250
n_chains = 2
options(mc.cores = 2)
begin_time <- proc.time()
policy_mod <- stan(file = "~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/stan_models/policy_model.stan",
                   data = policy_stan_input$stan_input,
                   iter = n_iter,
                   warmup = n_warmup,
                   chains = n_chains)
run_time <- proc.time() - begin_time
print(run_time) 

# Check diagnostics ----

# Means
theta_sum = summary(policy_mod, pars = "theta")$summary
beta_sum = summary(policy_mod, pars = "beta")$summary
gamma_sum = summary(policy_mod, pars = "gamma")$summary

# SDs and rho
summary(policy_mod, pars = "sigma_theta")$summary
summary(policy_mod, pars = "sigma_beta")$summary
summary(policy_mod, pars = "sigma_gamma")$summary
summary(policy_mod, pars = "rho")$summary

# Save output ----

theta_draws = extract(policy_mod, pars = "theta")$theta[1:300,,]
dimnames(theta_draws)[[2]] = policy_stan_input$L1_dimnames
names(attr(theta_draws, "dimnames"))[2:3] = c("state", "t_int")
str(theta_draws)

saveRDS(theta_draws, "./model_output/theta_draws.rds")
