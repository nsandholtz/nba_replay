library(dplyr)

expit = function(x){exp(x)/(1+exp(x))}

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
  