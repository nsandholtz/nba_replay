library(extraDistr)
source("./code/simulation_utils.R")

# Load data
dat = read.csv("./data/2013_11_01_MIA_BRK_formatted.csv")

########################################
#########  SIMULATOR INPUTS  ###########
########################################

# Get initial states and shot clock times
MIA_initial_states <- get_initial_states(dat, "MIA")
BRK_initial_states <- get_initial_states(dat, "BRK")

# Get empirical shot clock distribution, L()
shot_clock_dist <- get_sc_dist(dat = dat, num_intervals = 3)

# Load posterior draws
lambda_MIA_draws = readRDS("./model_output/lambda_MIA_draws.rds")
lambda_BRK_draws = readRDS("./model_output/lambda_BRK_draws.rds")
mu_draws = readRDS("./model_output/mu_draws.rds")
theta_draws = readRDS("./model_output/theta_draws.rds")
xi_draws = readRDS("./model_output/xi_draws.rds")
n_draws = 300


########################################
##########    ALGORITHM 1    ###########
########################################

algorithm_1 = function(s_0, 
                       theta_draws,
                       mu_draws,
                       xi_draws,
                       lambda_draws,
                       c_0,
                       L_dist,
                       num_mcmc){
  s_n = c_n = t_n = a_n = r_n = region = NA
  
  s_n[1] = s_0 
  c_n[1] = c_0
  
  n = 1
  while(s_n[n] != "turnover"){
    region = strsplit(s_n[n], split = "_")[[1]][2]
    draw_index = sample(1:num_mcmc,
                        size = 1)
    t_n[n] = get_t_int(c_n[n])
    a_n[n] = ifelse(region == "heave",
                    0,
                    rbinom(1, 1, prob = expit(theta_draws[draw_index,s_n[n],t_n[n]])))
    if(a_n[n] == 1){
      reward_state = paste(strsplit(s_n[n], split = "_")[[1]][1],
                           strsplit(s_n[n], split = "_")[[1]][2], sep = "_")
      
      r_n[n] = Reward(mu = mu_draws[draw_index, reward_state],
                      xi = xi_draws[draw_index, region],
                      def_open = strsplit(s_n[n], split = "_")[[1]][3] == "open",
                      region = region)
      break  
    } else {
      r_n[n] = 0
    }
    lapse = sample(L_dist[[t_n[n]]], 1)
    c_n[n+1] = c_n[n] - lapse
    if(c_n[n+1] < 0){
      s_n[n+1] = "turnover"
    } else {
      s_n[n+1] = dimnames(lambda_draws)[[3]][rcat(1, prob = softmax(lambda_draws[draw_index, s_n[n], , t_n[n]]))]
    }
    n = n + 1
  }
  if (s_n[n] == "turnover") {
    a_n[n] = NA
    r_n[n] = 0
    t_n[n] = get_t_int(c_n[n])
  }
  output = data.frame(state = s_n,
                      action = a_n, 
                      reward = r_n, 
                      shot_clock = c_n,
                      time_int = t_n,
                      play = play)
  return(output)
} 

########################################
#########   SIMULATE PLAYS   ###########
########################################

n_sim = 300

# MIAMI SIMULATIONS
MIA_points = NA
for(iter in 1:n_sim){
  cat(iter,"\r")
  for(play in 1:nrow(MIA_initial_states)) {
    if (play == 1) {
      game_moments_MIA = algorithm_1(
        s_0 = MIA_initial_states[play, "state"],
        c_0 = MIA_initial_states[play, "shot_clock"],
        theta_draws = theta_draws,
        mu_draws = mu_draws,
        xi_draws = xi_draws,
        lambda_draws = lambda_MIA_draws,
        L_dist = shot_clock_dist,
        num_mcmc = n_draws
      )
    } else {
      game_moments_MIA = rbind(
        game_moments_MIA,
        algorithm_1(
          s_0 = MIA_initial_states[play, "state"],
          c_0 = MIA_initial_states[play, "shot_clock"],
          theta_draws = theta_draws,
          mu_draws = mu_draws,
          xi_draws = xi_draws,
          lambda_draws = lambda_MIA_draws,
          L_dist = shot_clock_dist,
          num_mcmc = n_draws
        )
      )
    }
  }
  MIA_points[iter] = sum(game_moments_MIA$reward)
}

# BROOKLYN SIMULATIONS
BRK_points = NA
for(iter in 1:n_sim){
  cat(iter,"\r")
  for(play in 1:nrow(BRK_initial_states)) {
    if (play == 1) {
      game_moments_BRK = algorithm_1(
        s_0 = BRK_initial_states[play, "state"],
        c_0 = BRK_initial_states[play, "shot_clock"],
        theta_draws = theta_draws,
        mu_draws = mu_draws,
        xi_draws = xi_draws,
        lambda_draws = lambda_BRK_draws,
        L_dist = shot_clock_dist,
        num_mcmc = n_draws
      )
    } else {
      game_moments_BRK = rbind(
        game_moments_BRK,
        algorithm_1(
          s_0 = BRK_initial_states[play, "state"],
          c_0 = BRK_initial_states[play, "shot_clock"],
          theta_draws = theta_draws,
          mu_draws = mu_draws,
          xi_draws = xi_draws,
          lambda_draws = lambda_BRK_draws,
          L_dist = shot_clock_dist,
          num_mcmc = n_draws
        )
      )
    }
  }
  BRK_points[iter] = sum(game_moments_BRK$reward)
}

########################################
#########    PLOT RESULTS    ###########
########################################

# Compare simulations to empirical
plot(density(MIA_points), col = "red", main = "Simulations: MIA vs BRK")
dat %>% filter(team == "MIA") %>% with(abline(v = sum(points), 
                                              col = "red",
                                              lty = 2))

lines(density(BRK_points))
dat %>% filter(team == "BRK") %>% with(abline(v = sum(points),
                                              lty = 2))

########################################
#########   ALTERED POLICY   ###########
########################################

# Identify MIA players
MIA_players = dat %>%
  filter(team == "MIA") %>%
  distinct(entity) %>%
  pull(entity)

# Identify states to alter
# 1) ALL Midrange shots
to_alter_1 = c(paste(MIA_players, "long2_contested", sep = "_"),
             paste(MIA_players, "long2_open", sep = "_"))
# 2) ALL three point shots
to_alter_2 = c(paste(MIA_players, "three_contested", sep = "_"),
               paste(MIA_players, "three_open", sep = "_"))
  

# POLICY ALTERATION
# Decrease midrange shot policy by 20% (except late in shot clock) and 
# increase three point policy by 20% (regardless of time on clock)
conservative_policy_change <- list(list(who_where = to_alter_1,
                                        when = 2:3,
                                        how_much = .8),
                                   list(who_where = to_alter_2,
                                        when = 1:3,
                                        how_much = 1.2)
                                   )

# Alter the posterior draws of theta
altered_theta_draws = alter_theta(theta_draws, 
                                  altered_policy_rules = conservative_policy_change)

# MIAMI ALTERED SIMULATIONS
MIA_points_alt = NA
for(iter in 1:n_sim){
  cat(iter,"\r")
  for(play in 1:nrow(MIA_initial_states)) {
    if (play == 1) {
      game_moments_MIA = algorithm_1(
        s_0 = MIA_initial_states[play, "state"],
        c_0 = MIA_initial_states[play, "shot_clock"],
        theta_draws = altered_theta_draws,
        mu_draws = mu_draws,
        xi_draws = xi_draws,
        lambda_draws = lambda_MIA_draws,
        L_dist = shot_clock_dist,
        num_mcmc = n_draws
      )
    } else {
      game_moments_MIA = rbind(
        game_moments_MIA,
        algorithm_1(
          s_0 = MIA_initial_states[play, "state"],
          c_0 = MIA_initial_states[play, "shot_clock"],
          theta_draws = altered_theta_draws,
          mu_draws = mu_draws,
          xi_draws = xi_draws,
          lambda_draws = lambda_MIA_draws,
          L_dist = shot_clock_dist,
          num_mcmc = n_draws
        )
      )
    }
  }
  MIA_points_alt[iter] = sum(game_moments_MIA$reward)
}

lines(density(MIA_points_alt), col = "blue")
# Miami's projected distribution of possible scores increases


