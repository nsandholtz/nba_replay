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

dat_policy = dat %>%
  filter(location_id != 'heave') %>%
  mutate(entity = as.factor(as.character(entity)),
         court_region = as.character(location_id),
         court_region = ifelse(court_region %in% c('arc3', 'corner3'),
                               'three',
                               ifelse(court_region == c('dunk'), 
                                      'paint', court_region)),
         court_region = as.factor(as.character(court_region)),
         player_region_defense = as.factor(paste(entity, 
                                       court_region,
                                       def_pres, 
                                       sep = "_")),
         position_region_defense = as.factor(paste(position_simple, 
                                       court_region,
                                       def_pres, 
                                       sep = "_")),
         region_defense = as.factor(paste(court_region,
                                def_pres,
                                sep = "_")),
         time_int = if_else(shot_clock <= 8, 1,
                            if_else(shot_clock > 8  & shot_clock <= 16, 2, 
                                    3)))

player_positions <- dat_policy %>% 
  group_by(entity) %>% summarize(group = first(position_simple),
                                 pos = as.numeric(as.factor(first(position_simple)))) %>%
  arrange(entity)

# Stan inputs ----

policy_stan_input_simplified <- list(
  # Defining dimensions
  N_obs = nrow(dat_policy),
  T_int = length(unique(dat_policy$time_int)),
  P = length(unique(dat_policy$entity)),
  G = length(unique(dat_policy$position_simple)),
  R = length(unique(dat_policy$court_region)),
  K = length(unique(dat_policy$region_defense)),
  # Raw data
  A = ifelse(dat_policy$event_id %in% c(3,4), 1, 0),
  A_states = as.numeric(dat_policy$region_defense),  
  A_players = as.numeric(dat_policy$entity),  
  A_regions = as.numeric(dat_policy$court_region),  
  A_time = dat_policy$time_int,
  # Indexing players
  player_position = player_positions$pos,
  # Final-stage (Hyperprior) fixed values
  gamma_mean = rep(0, length(unique(dat_policy$time_int)))
)

# Fit model ----

options(mc.cores = 2)
begin_time <- proc.time()
policy_mod <- stan(file = "~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/stan_models/policy_model_simplified.stan",
                   data = policy_stan_input_simplified,
                   iter = 200,
                   warmup = 50,
                   chains = 2)
run_time <- proc.time() - begin_time
print(run_time) 

# Check diagnostics ----

# Means
View(summary(policy_mod, pars = "theta")$summary)
View(summary(policy_mod, pars = "beta")$summary)
View(summary(policy_mod, pars = "gamma")$summary)

# SDs
summary(policy_mod, pars = "sigma_theta")$summary
summary(policy_mod, pars = "sigma_beta")$summary
summary(policy_mod, pars = "sigma_gamma")$summary

# Transformed mu - make probabilities

View(summary(policy_mod, pars = "theta_trans")$summary)
View(summary(policy_mod, pars = "theta_trans")$summary)

# Need to do something about saving the output for the simulation
