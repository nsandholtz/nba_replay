library(dplyr)
library(rstan)

# Get data ready ----
dat = read.csv("./data/2013_11_01_MIA_BRK_formatted.csv")

# Create a column denoting the player_region state,
# a column for the first stage of the hierarchical prior,
# a column for the second stage of the hierarchical prior.

# For simplicity in this one-game tutorial, 
# we do not estimate the make probability of 'heaves',
# and we use the same naive player positions as in the 
# policy model.  

dat_reward = dat %>%
  filter(location_id != 'heave',
         event_id %in% c(3,4)) %>%
  mutate(entity = as.factor(as.character(entity)),
         court_region = as.factor(as.character(location_id)),
         player_region = as.factor(paste(entity,
                                         court_region,
                                         sep = "_")), 
         position_region = as.factor(paste(position_simple, 
                                           court_region,
                                           sep = "_")))


player_positions <- dat_reward %>% 
  group_by(entity) %>% summarize(group = first(position_simple),
                                 pos = as.numeric(as.factor(first(position_simple)))) %>%
  arrange(entity)

# Stan inputs ----

reward_stan_input <- list(
  # Defining dimensions
  N_obs = nrow(dat_reward),
  P = length(unique(dat_reward$entity)),
  H = length(unique(dat_reward$position_simple)),
  R = length(unique(dat_reward$court_region)),
  # Raw data
  M = ifelse(dat_reward$event_id == 3, 1, 0),
  M_player = as.numeric(dat_reward$entity),  
  M_group = as.numeric(dat_reward$position_simple),  
  M_region = as.numeric(dat_reward$court_region),  
  M_contested = ifelse(dat_reward$def_pres == 'contested', 1, 0),
  # Indexing players
  player_group = player_positions$pos
  # Final-stage (Hyperprior) fixed values
)

# Fit model ----

options(mc.cores = 2)
begin_time <- proc.time()
reward_mod <- stan(file = "~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/stan_models/reward_model.stan",
                   data = reward_stan_input,
                   iter = 20000,
                   warmup = 5000,
                   chains = 2)
run_time <- proc.time() - begin_time
print(run_time) 

# Need to do something about saving the output for the simulation

# Means
View(summary(reward_mod, pars = "mu")$summary)
View(summary(reward_mod, pars = "psi")$summary)
View(summary(reward_mod, pars = "varphi")$summary)
summary(reward_mod, pars = "xi")$summary

# SDs
summary(reward_mod, pars = "sigma_mu")$summary
summary(reward_mod, pars = "sigma_psi")$summary
summary(reward_mod, pars = "sigma_varphi")$summary
summary(reward_mod, pars = "sigma_xi")$summary

