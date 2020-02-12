library(dplyr)
library(rstan)

# Get data ready ----
dat = read.csv("./data/2013_11_01_MIA_BRK_formatted.csv")

# Create a column denoting the state space of the MDP,
# a column for the first stage of the hierarchical prior,
# a column for the second stage of the hierarchical prior,
# a column for interval of the shot clock (3 intervals)

# For simplicity in this one-game tutorial
# we use only three player positions --- (G,F,C).
# We also simplify the court regions to three levels --- (paint, long2, three)

dat_transition = dat %>%
  mutate(event_next = lead(event_id),
         entity = factor(as.character(entity)),
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
         player_region_defense_next = lead(player_region_defense),
         position_region_defense_next = lead(position_region_defense),
         region_defense_next = lead(region_defense),
         time_int = if_else(shot_clock <= 8, 1,
                            if_else(shot_clock > 8  & shot_clock <= 16, 2, 
                                    3)),
         player_region_defense_next = if_else(event_next == 7, 
                                              "turnover", 
                                              as.character(player_region_defense_next)),
         player_region_defense_next = factor(player_region_defense_next,
                                             levels = c(levels(player_region_defense),"turnover")),
         position_region_defense_next = if_else(event_next == 7, 
                                              "turnover", 
                                              as.character(position_region_defense_next)),
         position_region_defense_next = factor(position_region_defense_next,
                                             levels = c(levels(position_region_defense),"turnover")),
         region_defense_next = if_else(event_next == 7, 
                                                "turnover", 
                                                as.character(region_defense_next)),
         region_defense_next = factor(region_defense_next,
                                               levels = c(levels(region_defense),"turnover"))) %>%
  filter(!(event_id %in% c(3,4,7))) %>%
  select(play,
         event_id,
         event_next,
         player_region_defense,
         player_region_defense_next,
         position_region_defense,
         position_region_defense_next,
         region_defense,
         region_defense_next,
         time_int)

# Getting indexes for stan ----
player_positions <- dat %>% 
  mutate(entity = factor(as.character(entity))) %>%
  group_by(entity) %>% summarize(group = first(position_simple),
                                 pos = as.numeric(as.factor(first(position_simple)))) %>%
  arrange(entity)

L1_levels = levels(dat_transition$player_region_defense_next)
L2_levels = levels(dat_transition$position_region_defense_next)
L3_levels = levels(dat_transition$region_defense_next)

L1_to_L2 = NA
for(i in 1:length(L1_levels)){
  matching_tool_1 = strsplit(L1_levels[i], "_")[[1]]
  if(matching_tool_1[1] == "turnover"){
    L1_to_L2[i] = match(matching_tool_1, L2_levels)
  } else {
    state_position = as.character(player_positions$group[match(matching_tool_1[1],player_positions$entity)])
    matching_tool_2 = paste(state_position, matching_tool_1[2], matching_tool_1[3], sep = "_")
    L1_to_L2[i] = match(matching_tool_2, L2_levels)
  }
}

L2_to_L3 = NA
for(i in 1:length(L2_levels)){
  if(L2_levels[i] == "turnover"){
    L2_to_L3[i] = match(L2_levels[i], L3_levels)
  } else {
    L2_to_L3[i] = match(substr(L2_levels[i], 3, nchar(L2_levels[i])), L3_levels)
  }
}

# Stan inputs ----

transition_stan_input <- list(
  # Defining dimensions
  N_obs = nrow(dat_transition),
  T_int = length(unique(dat_transition$time_int)),
  L1_r = length(levels(dat_transition$player_region_defense)),
  L1_c = length(levels(dat_transition$player_region_defense_next)),
  L2_r = length(levels(dat_transition$position_region_defense)),
  L2_c = length(levels(dat_transition$position_region_defense_next)),
  L3_r = length(levels(dat_transition$region_defense)),
  L3_c = length(levels(dat_transition$region_defense_next)),
  # Raw data
  S_next = as.numeric(dat_transition$player_region_defense_next), 
  S_orig = as.numeric(dat_transition$player_region_defense),  
  S_time = dat_transition$time_int,
  # Connecting hierarchies
  L1_to_L2 = L1_to_L2,
  L2_to_L3 = L2_to_L3,
  # Final-stage (Hyperprior) fixed values
  omega_mean = rep(0, length(unique(dat_transition$time_int))),
  half_cauchy_scale = 2.5
)

# Fit model ----

n_iter = 2000
n_warmup = 500
n_chains = 2
options(mc.cores = 2)
begin_time <- proc.time()
transition_mod <- stan(file = "~/Dropbox/Luke_Research/Shot_Policy/nba_replay/code/stan_models/transition_probability_model.stan",
                   data = transition_stan_input,
                   iter = n_iter,
                   warmup = n_warmup,
                   chains = n_chains)
run_time <- proc.time() - begin_time
print(run_time) 

# Check diagnostics ----

# Means
lambda_sum = summary(transition_mod, pars = "lambda")$summary
zeta_sum = summary(transition_mod, pars = "zeta")$summary
omega_sum = summary(transition_mod, pars = "omega")$summary

# SDs and rho
summary(transition_mod, pars = "sigma_lambda")$summary
summary(transition_mod, pars = "sigma_zeta")$summary
summary(transition_mod, pars = "sigma_omega")$summary
summary(transition_mod, pars = "rho")$summary

# Saving the output, which is used in the simulation

lambda_draws = extract(transition_mod, pars = "lambda")$lambda[1:150,,,]
saveRDS(lambda_draws, "./model_output/lambda_draws.rds")
