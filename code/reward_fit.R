library(dplyr)
library(rstan)

# Get data ready ----
dat = read.csv("./data/2013_11_01_MIA_BRK_formatted.csv")

# Create a column denoting the player_region state,
# a column for the first stage of the hierarchical prior,
# a column for the second stage of the hierarchical prior.

# For simplicity in this one-game tutorial, 
# we do not estimate the make probability of 'heaves',
# and we use the same player positions as in the 
# policy model.  We also simplify the court regions
# to three levels --- (paint, long2, and three)

# Define factor levels
player_region_levels = dat %>% 
  mutate(entity = as.factor(as.character(entity)),
         court_region = as.character(location_id),
         court_region = ifelse(court_region %in% c('arc3', 'corner3'),
                               'three',
                               ifelse(court_region == c('dunk'), 
                                      'paint', court_region)),
         court_region = as.factor(as.character(court_region)),
         player_region = interaction(entity,
                                     court_region,
                                     sep = "_",
                                     lex.order = T)) %>%
  with(levels(player_region))

# Remove heaves from levels
heave_indexes = which(unlist(lapply(strsplit(player_region_levels, "_"), function(x) x[2])) == "heave")
levels_no_heave = player_region_levels[-heave_indexes]

# Filter data applicable for reward model
dat_reward = dat %>%
  filter(location_id != 'heave',
         event_id %in% c(3,4)) %>%
  mutate(entity = as.factor(as.character(entity)),
         court_region = as.character(location_id),
         court_region = ifelse(court_region %in% c('arc3', 'corner3'),
                               'three',
                               ifelse(court_region == c('dunk'), 
                                      'paint', court_region)),
         court_region = as.factor(as.character(court_region)),
         player_region = factor(paste(as.character(entity),
                                      as.character(court_region),
                                     sep = "_"),
                                levels = levels_no_heave),
         position_region = interaction(position_simple, 
                                           court_region,
                                           sep = "_"))

# Getting indexes for stan ----

player_positions <- dat %>% 
  mutate(entity = factor(as.character(entity))) %>%
  group_by(entity) %>% summarize(group = first(position_simple),
                                 pos = as.numeric(as.factor(first(position_simple)))) %>%
  arrange(entity)

L1_levels = levels(dat_reward$player_region)
L2_levels = levels(dat_reward$position_region)
L3_levels = levels(dat_reward$court_region)

L1_to_L2 = NA
for(i in 1:length(L1_levels)){
  matching_tool_1 = strsplit(L1_levels[i], "_")[[1]]
  state_position = as.character(player_positions$group[match(matching_tool_1[1],player_positions$entity)])
  matching_tool_2 = paste(state_position, matching_tool_1[2], sep = "_")
  L1_to_L2[i] = match(matching_tool_2, L2_levels)
}

L2_to_L3 = NA
for(i in 1:length(L2_levels)){
  L2_to_L3[i] = match(substr(L2_levels[i], 3, nchar(L2_levels[i])), L3_levels)
}

# Stan inputs ----

reward_stan_input <- list(
  # Defining dimensions
  N_obs = nrow(dat_reward),
  L1 = length(L1_levels),
  L2 = length(L2_levels),
  L3 = length(L3_levels),
  # Raw data
  M = ifelse(dat_reward$event_id == 3, 1, 0),
  M_player_region = as.numeric(dat_reward$player_region),  
  M_region = as.numeric(dat_reward$court_region),  
  M_open = ifelse(dat_reward$def_pres == 'open', 1, 0),
  # Connecting hierarchy indexes
  L1_to_L2 = L1_to_L2,
  L2_to_L3 = L2_to_L3,
  # Final-stage (Hyperprior) fixed values
  varphi_mean = 0,
  half_cauchy_scale = 2.5
)

# Fit model ----

n_iter = 2000
n_warmup = 500
n_chains = 2
options(mc.cores = 2)
begin_time <- proc.time()
reward_mod <- stan(file = "~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/stan_models/reward_model.stan",
                   data = reward_stan_input,
                   iter = n_iter,
                   warmup = n_warmup,
                   chains = n_chains)
run_time <- proc.time() - begin_time
print(run_time) 

# Check diagnostics ----

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

# Saving output for simulation
mu_draws = extract(reward_mod, pars = "mu")$mu
xi_draws = extract(reward_mod, pars = "xi")$xi

saveRDS(mu_draws, "./model_output/mu_draws.rds")
saveRDS(xi_draws, "./model_output/xi_draws.rds")
