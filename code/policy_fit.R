library(dplyr)
library(rstan)

# Get data ready ----
dat = read.csv("./data/2013_11_01_MIA_BRK_formatted.csv")

# Create a column denoting the state space of the MDP,
# a column for the first stage of the hierarchical prior,
# a column for the second stage of the hierarchical prior,
# a column for interval of the shot clock (3 intervals)

dat_policy = dat %>%
  mutate(entity = as.factor(as.character(entity)),
         player_region_defense = as.factor(paste(entity, 
                                       location_id,
                                       def_pres, 
                                       sep = "_")),
         position_region_defense = as.factor(paste(position_simple, 
                                       location_id,
                                       def_pres, 
                                       sep = "_")),
         region_defense = as.factor(paste(location_id,
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

policy_stan_input <- list(
  # Defining dimensions
  N_obs = nrow(dat_policy),
  T_int = length(unique(dat_policy$time_int)),
  P = length(unique(dat_policy$entity)),
  G = length(unique(dat_policy$position_simple)),
  K = length(unique(dat_policy$region_defense)),
  # Raw data
  A = ifelse(dat_policy$event_id %in% c(3,4), 1, 0),
  A_states = as.numeric(dat_policy$region_defense),  
  A_players = as.numeric(dat_policy$entity),  
  A_time = dat_policy$time_int,
  # Indexing players
  player_position = player_positions$pos,
  # Final-stage (Hyperprior) fixed values
  gamma_mean = rep(0, length(unique(dat_policy$time_int)))
)

# Fit model ----

options(mc.cores = 2)
begin_time <- proc.time()
policy_mod <- stan(file = "~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/policy_model.stan",
                         data = policy_stan_input,
                         iter = 2000,
                         warmup = 500,
                         chains = 2)
run_time <- proc.time() - begin_time
print(run_time) #1772.130 seconds

# ALTERNATE METHOD TO SPEED UP COMPUTATION USING THE BINOMIAL DISTRIBUTION

source("~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/utils.R")
transition_count_matrix <- get_policy_transitions(dat_policy,
                                                  state_col = "region_defense",
                                                  num_intervals = 3,
                                                  TPM_or_TCM = "TCM")





