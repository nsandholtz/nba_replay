library(dplyr)
library(rstan)

# Load data and utils ----

source("./code/model_fitting_utils.R")
dat = read.csv("./data/2013_11_01_MIA_BRK_formatted.csv")

# JOINT FIT ---------------------------------------------------------------

transition_stan_input = get_transition_model_inputs(dat,
                                                    my_teams = c("MIA","BRK"))

# Fit joint model ----
n_iter = 300
n_warmup = 150
n_chains = 2
options(mc.cores = 2)
begin_time <- proc.time()
transition_mod <- stan(file = "~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/stan_models/transition_probability_model_joint.stan",
                           data = transition_stan_input$stan_input,
                           iter = n_iter,
                           warmup = n_warmup,
                           chains = n_chains)
run_time <- proc.time() - begin_time
print(run_time) 

lambda_MIA_draws = extract(transition_mod, pars = "lambda_team_1")$lambda_team_1[1:300,,,]
dimnames(lambda_MIA_draws)[[2]] = joint_transition_stan_input$L1_dimnames[[1]][-length(joint_transition_stan_input$L1_dimnames[[1]])]
dimnames(lambda_MIA_draws)[[3]] = joint_transition_stan_input$L1_dimnames[[1]]
names(attr(lambda_MIA_draws, "dimnames"))[2:4] = c("orig_state", "dest_state", "t_int")
saveRDS(lambda_MIA_draws, "./model_output/lambda_MIA_draws.rds")

lambda_BRK_draws = extract(transition_mod, pars = "lambda_team_2")$lambda_team_2[1:300,,,]
dimnames(lambda_BRK_draws)[[2]] = joint_transition_stan_input$L1_dimnames[[2]][-length(joint_transition_stan_input$L1_dimnames[[2]])]
dimnames(lambda_BRK_draws)[[3]] = joint_transition_stan_input$L1_dimnames[[2]]
names(attr(lambda_BRK_draws, "dimnames"))[2:4] = c("orig_state", "dest_state", "t_int")
saveRDS(lambda_BRK_draws, "./model_output/lambda_BRK_draws.rds")
str(lambda_BRK_draws)
