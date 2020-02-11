library(dplyr)
library(rstan)

# Get data ready ----
dat = read.csv("./data/2013_11_01_MIA_BRK_formatted.csv")

# Create a column denoting the state space of the MDP,
# a column for the first stage of the hierarchical prior,
# a column for the second stage of the hierarchical prior,
# a column for interval of the shot clock (3 intervals)

# For simplicity in this one-game tutorial, 
# we do not estimate the shooting probability of 'heaves',
# and we use only three player positions --- (G,F,C).
# We also simplify the court regions to three levels --- (paint, long2, three)

# Define levels first

dat_policy = dat %>%
  filter(location_id != 'heave') %>%
  mutate(entity = factor(as.character(entity)),
         court_region = as.character(location_id),
         court_region = ifelse(court_region %in% c('arc3', 'corner3'),
                               'three',
                               ifelse(court_region == c('dunk'), 
                                      'paint', court_region)),
         court_region = factor(as.character(court_region)),
         player_region_defense = interaction(entity, 
                                       court_region,
                                       def_pres, 
                                       sep = "_", lex.order = T),
         position_region_defense = interaction(position_simple, 
                                       court_region,
                                       def_pres, 
                                       sep = "_", lex.order = T),
         region_defense = interaction(court_region,
                                def_pres,
                                sep = "_", lex.order = T),
         time_int = if_else(shot_clock <= 8, 1,
                            if_else(shot_clock > 8  & shot_clock <= 16, 2, 
                                    3)))

# Getting indexes for stan ----
player_positions <- dat %>% 
  mutate(entity = factor(as.character(entity))) %>%
  group_by(entity) %>% summarize(group = first(position_simple),
                                 pos = as.numeric(as.factor(first(position_simple)))) %>%
  arrange(entity)

A_state_L1 = levels(dat_policy$player_region_defense)
A_state_L2 = levels(dat_policy$position_region_defense)
A_state_L3 = levels(dat_policy$region_defense)

L1_to_L2 = NA
for(i in 1:length(A_state_L1)){
  matching_tool_1 = strsplit(A_state_L1[i], "_")[[1]]
  state_position = as.character(player_positions$group[match(matching_tool_1[1],player_positions$entity)])
  matching_tool_2 = paste(state_position, matching_tool_1[2], matching_tool_1[3], sep = "_")
  L1_to_L2[i] = match(matching_tool_2, A_state_L2)
}

L2_to_L3 = NA
for(i in 1:length(A_state_L2)){
  L2_to_L3[i] = match(substr(A_state_L2[i], 3, nchar(A_state_L2[i])), A_state_L3)
}

# Stan inputs ----

policy_stan_input <- list(
  # Defining dimensions
  N_obs = nrow(dat_policy),
  T_int = length(unique(dat_policy$time_int)),
  L1 = length(A_state_L1),
  L2 = length(A_state_L2),
  L3 = length(A_state_L3),
  # Raw data
  A = ifelse(dat_policy$event_id %in% c(3,4), 1, 0),
  A_state = as.numeric(dat_policy$player_region_defense),  
  A_time = dat_policy$time_int,
  # Indexing players
  L1_to_L2 = L1_to_L2,
  L2_to_L3 = L2_to_L3,
  # Final-stage (Hyperprior) fixed values
  gamma_mean = rep(0, length(unique(dat_policy$time_int)))
)

# Fit model ----

options(mc.cores = 2)
begin_time <- proc.time()
policy_mod <- stan(file = "~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/stan_models/policy_model.stan",
                   data = policy_stan_input,
                   iter = 2000,
                   warmup = 500,
                   chains = 2)
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

# Results ----

# Compare model estimates to empirical probabilities

# Model estimates --- inverse logit transform
theta_prob = data.frame(matrix(exp(theta_sum[,1])/(1 + exp(theta_sum[,1])), length(A_state_L1), 3),
                        row.names = A_state_L1)
beta_prob = data.frame(matrix(exp(beta_sum[,1])/(1 + exp(beta_sum[,1])), length(A_state_L2), 3),
                       row.names = A_state_L2)
gamma_prob = data.frame(matrix(exp(gamma_sum[,1])/(1 + exp(gamma_sum[,1])), length(A_state_L3), 3),
                        row.names = A_state_L3)
colnames(theta_prob) = colnames(beta_prob) = colnames(gamma_prob) = c('0-8','8-16','16-24')

# Empirical probabilities

beta_empirical_num = dat_policy %>%
  filter(event_id %in% c(3,4)) %>%
  with(table(position_region_defense, time_int))

beta_empirical_denom = dat_policy %>%
  with(table(position_region_defense, time_int))

gamma_empirical_num = dat_policy %>%
  filter(event_id %in% c(3,4)) %>%
  with(table(region_defense, time_int))

gamma_empirical_denom = dat_policy %>%
  with(table(region_defense, time_int))

# Comparison

beta_prob
beta_empirical_num/beta_empirical_denom

gamma_prob
gamma_empirical_num/gamma_empirical_denom

# The shrinkage/borrowing strength from lower levels of 
# the hierarchy is strongly evident.  This is to be
# expected given that we are making estimates based on
# a single game of data.  

# Saving the output, which is used in the simulation

saveRDS(policy_mod, "./model_output/policy_fit.rds")
