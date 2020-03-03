library(dplyr)
library(extraDistr)

expit = function(x){exp(x)/(1+exp(x))}
logit = function(p){log(p/(1-p))}

softmax = function(x){exp(x)/sum(exp(x))}

Reward = function(mu, xi, def_open, region){
  points = ifelse(region == "three",
                  3 * expit(mu + def_open * xi),
                  2 * expit(mu + def_open * xi))
  return(points)
}

get_initial_states <- function(dat, my_team){
  initial_states = dat %>%
    mutate(is_first = c(1,diff(dat$play)) > 0,
           court_region = as.character(location_id),
           court_region = ifelse(court_region %in% c('arc3', 'corner3'),
                                 'three',
                                 ifelse(court_region == c('dunk'), 
                                        'paint', court_region)),
           state = paste(entity,
                         court_region,
                         def_pres,
                         sep = "_")) %>%
    filter(is_first == TRUE,
           team == my_team) %>%
    select(state, shot_clock)
  return(initial_states)
}

get_sc_dist <- function(dat, num_intervals = 3
                       # max_diff = -24, 
                        #max_diff_no_time = -.5, 
                        #action_type = "non_shots"
                       ){
  shot_clock_dist <- list()
  sc_partition <- seq(0,24,length.out = num_intervals + 1)
  for(i in 1:num_intervals) {
    shot_clock_dist[[i]]  = dat %>%
      filter(shot_clock > sc_partition[i],
             shot_clock <= sc_partition[i+1],
             !is.na(time_lapse),
             time_lapse < 0) %>%
      mutate(time_lapse = abs(time_lapse)) %>%
      pull(time_lapse)
  }
  return(shot_clock_dist)
}
  
get_t_int = function(c_n, num_intervals = 3){
  sc_partition <- seq(0,24,length.out = num_intervals + 1)
  t_n = NA
  for(i in length(sc_partition):2){
    if(c_n <= sc_partition[i] & 
       c_n > sc_partition[i-1]){
      t_n = i - 1
      break
    } 
  }
  return(t_n)
}
  
alter_theta = function(theta_draws,
                       altered_policy_rules,
                       threshold = .9){
  altered_theta_draws_probs = expit(theta_draws)
    for(i in 1:length(altered_policy_rules)){
      affected_inds = NULL
      for(j in 1:length(altered_policy_rules[[i]]$who_where)){
        affected_inds <- grep(altered_policy_rules[[i]]$who_where[j],dimnames(theta_draws)[[2]])
        altered_theta_draws_probs[, affected_inds,altered_policy_rules[[i]]$when] <- altered_theta_draws_probs[, affected_inds,altered_policy_rules[[i]]$when]*altered_policy_rules[[i]]$how_much
        altered_theta_draws_probs[, affected_inds,altered_policy_rules[[i]]$when][altered_theta_draws_probs[ , affected_inds,altered_policy_rules[[i]]$when] > threshold] <- threshold
      }
    }
  return(logit(altered_theta_draws_probs))
}

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
