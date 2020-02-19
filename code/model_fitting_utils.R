library(dplyr)

# Function to get policy model inputs
get_policy_inputs = function(dat, 
                             half_cauchy_scale = 2.5){
  
  # Create a column denoting the state space of the MDP,
  # a column for the first stage of the hierarchical prior,
  # a column for the second stage of the hierarchical prior,
  # a column for interval of the shot clock (3 intervals)
  
  # For simplicity in this one-game tutorial, 
  # we do not estimate the shooting probability of 'heaves',
  # and we use only three player positions --- (G,F,C).
  # We also simplify the court regions to three levels --- (paint, long2, three)
  # May want to define levels first
  
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
  
  L1_levels = levels(dat_policy$player_region_defense)
  L2_levels = levels(dat_policy$position_region_defense)
  L3_levels = levels(dat_policy$region_defense)
  
  L1_to_L2 = NA
  for(i in 1:length(L1_levels)){
    matching_tool_1 = strsplit(L1_levels[i], "_")[[1]]
    state_position = as.character(player_positions$group[match(matching_tool_1[1],player_positions$entity)])
    matching_tool_2 = paste(state_position, matching_tool_1[2], matching_tool_1[3], sep = "_")
    L1_to_L2[i] = match(matching_tool_2, L2_levels)
  }
  
  L2_to_L3 = NA
  for(i in 1:length(L2_levels)){
    L2_to_L3[i] = match(substr(L2_levels[i], 3, nchar(L2_levels[i])), L3_levels)
  }
  
  # Stan inputs ----
  
  policy_stan_input <- list(
    # Defining dimensions
    N_obs = nrow(dat_policy),
    T_int = length(unique(dat_policy$time_int)),
    L1 = length(L1_levels),
    L2 = length(L2_levels),
    L3 = length(L3_levels),
    # Raw data
    A = ifelse(dat_policy$event_id %in% c(3,4), 1, 0),
    A_state = as.numeric(dat_policy$player_region_defense),  
    A_time = dat_policy$time_int,
    # Connecting hierarchy indexes
    L1_to_L2 = L1_to_L2,
    L2_to_L3 = L2_to_L3,
    # Final-stage (Hyperprior) fixed values
    gamma_mean = rep(0, length(unique(dat_policy$time_int))),
    half_cauchy_scale = half_cauchy_scale
  )
  return(list(stan_input = policy_stan_input,
              L1_dimnames = L1_levels,
              L2_dimnames = L2_levels,
              L3_dimnames = L3_levels))
}


# Function to get reward model inputs 
get_reward_inputs = function(dat,
                             half_cauchy_scale = 2.5) {
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
    mutate(
      entity = as.factor(as.character(entity)),
      court_region = as.character(location_id),
      court_region = ifelse(
        court_region %in% c('arc3', 'corner3'),
        'three',
        ifelse(court_region == c('dunk'),
               'paint', court_region)
      ),
      court_region = as.factor(as.character(court_region)),
      player_region = interaction(entity,
                                  court_region,
                                  sep = "_",
                                  lex.order = T)
    ) %>%
    with(levels(player_region))
  
  # Remove heaves from levels
  heave_indexes = which(unlist(lapply(strsplit(player_region_levels, "_"), function(x)
    x[2])) == "heave")
  levels_no_heave = player_region_levels[-heave_indexes]
  
  # Filter data applicable for reward model
  dat_reward = dat %>%
    filter(location_id != 'heave',
           event_id %in% c(3, 4)) %>%
    mutate(
      entity = as.factor(as.character(entity)),
      court_region = as.character(location_id),
      court_region = ifelse(
        court_region %in% c('arc3', 'corner3'),
        'three',
        ifelse(court_region == c('dunk'),
               'paint', court_region)
      ),
      court_region = as.factor(as.character(court_region)),
      player_region = factor(paste(
        as.character(entity),
        as.character(court_region),
        sep = "_"
      ),
      levels = levels_no_heave),
      position_region = interaction(position_simple,
                                    court_region,
                                    sep = "_")
    )
  
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
  for (i in 1:length(L1_levels)) {
    matching_tool_1 = strsplit(L1_levels[i], "_")[[1]]
    state_position = as.character(player_positions$group[match(matching_tool_1[1], player_positions$entity)])
    matching_tool_2 = paste(state_position, matching_tool_1[2], sep = "_")
    L1_to_L2[i] = match(matching_tool_2, L2_levels)
  }
  
  L2_to_L3 = NA
  for (i in 1:length(L2_levels)) {
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
  return(list(stan_input = reward_stan_input,
              L1_dimnames = L1_levels,
              L2_dimnames = L2_levels,
              L3_dimnames = L3_levels))
}

# Function to get joint transition probability model inputs
get_transition_model_inputs = function(dat,
                                       half_cauchy_scale = 2.5,
                                       my_teams) {
  # Create a column denoting the state space of the MDP,
  # a column for the first stage of the hierarchical prior,
  # a column for the second stage of the hierarchical prior,
  # a column for interval of the shot clock (3 intervals)
  
  # For simplicity in this one-game tutorial
  # we use only three player positions --- (G,F,C).
  # We also simplify the court regions to three levels --- (paint, long2, three)
  
  dat_transition = list()
  player_positions = list()
  L1_levels = list()
  L1_to_L2 = list()
  
  for(t in 1:length(my_teams)){
    dat_transition[[t]] = dat %>%
      filter(team == my_teams[t]) %>%
      mutate(
        event_next = lead(event_id),
        entity = factor(as.character(entity)),
        court_region = as.character(location_id),
        court_region = ifelse(
          court_region %in% c('arc3', 'corner3'),
          'three',
          ifelse(court_region == c('dunk'),
                 'paint', court_region)
        ),
        court_region = factor(as.character(court_region)),
        player_region_defense = interaction(
          entity,
          court_region,
          def_pres,
          sep = "_",
          lex.order = T
        ),
        position_region_defense = interaction(
          position_simple,
          court_region,
          def_pres,
          sep = "_",
          lex.order = T
        ),
        region_defense = interaction(
          court_region,
          def_pres,
          sep = "_",
          lex.order = T
        ),
        player_region_defense_next = lead(player_region_defense),
        position_region_defense_next = lead(position_region_defense),
        region_defense_next = lead(region_defense),
        time_int = if_else(shot_clock <= 8, 1,
                           if_else(shot_clock > 8  &
                                     shot_clock <= 16, 2,
                                   3)),
        player_region_defense_next = if_else(
          event_next == 7,
          "turnover",
          as.character(player_region_defense_next)
        ),
        player_region_defense_next = factor(player_region_defense_next,
                                            levels = c(
                                              levels(player_region_defense), "turnover"
                                            )),
        position_region_defense_next = if_else(
          event_next == 7,
          "turnover",
          as.character(position_region_defense_next)
        ),
        position_region_defense_next = factor(position_region_defense_next,
                                              levels = c(
                                                levels(position_region_defense), "turnover"
                                              )),
        region_defense_next = if_else(
          event_next == 7,
          "turnover",
          as.character(region_defense_next)
        ),
        region_defense_next = factor(region_defense_next,
                                     levels = c(levels(region_defense), "turnover"))
      ) %>%
      filter(!(event_id %in% c(3, 4, 7))) %>%
      select(
        play,
        event_id,
        event_next,
        player_region_defense,
        player_region_defense_next,
        position_region_defense,
        position_region_defense_next,
        region_defense,
        region_defense_next,
        time_int
      )
    
    # Getting indexes for stan ----
    player_positions[[t]] <- dat %>%
      filter(team == my_teams[t]) %>%
      mutate(entity = factor(as.character(entity))) %>%
      group_by(entity) %>% summarize(group = first(position_simple),
                                     pos = as.numeric(as.factor(first(position_simple)))) %>%
      arrange(entity)
    
    L1_levels[[t]] = levels(dat_transition[[t]]$player_region_defense_next)
    L2_levels = levels(dat_transition[[t]]$position_region_defense_next)
    L3_levels = levels(dat_transition[[t]]$region_defense_next)
    
    L1_to_L2[[t]] = NA
    for (i in 1:length(L1_levels[[t]])) {
      matching_tool_1 = strsplit(L1_levels[[t]][i], "_")[[1]]
      if (matching_tool_1[1] == "turnover") {
        L1_to_L2[[t]][i] = match(matching_tool_1, L2_levels)
      } else {
        state_position = as.character(player_positions[[t]]$group[match(matching_tool_1[1], player_positions[[t]]$entity)])
        matching_tool_2 = paste(state_position,
                                matching_tool_1[2],
                                matching_tool_1[3],
                                sep = "_")
        L1_to_L2[[t]][i] = match(matching_tool_2, L2_levels)
      }
    }
    
    L2_to_L3 = NA
    for (i in 1:length(L2_levels)) {
      if (L2_levels[i] == "turnover") {
        L2_to_L3[i] = match(L2_levels[i], L3_levels)
      } else {
        L2_to_L3[i] = match(substr(L2_levels[i], 3, nchar(L2_levels[i])), L3_levels)
      }
    }
  }
 
  # Stan inputs ----
  
  transition_stan_input <- list(
    # Defining dimensions
    N_teams = length(my_teams),
    N_obs = c(nrow(dat_transition[[1]]),
              nrow(dat_transition[[2]])),
    T_int = length(unique(dat_transition[[1]]$time_int)),
    L1_r = unlist(lapply(dat_transition, function(x) length(levels(x$player_region_defense)))),
    L1_c = unlist(lapply(dat_transition, function(x) length(levels(x$player_region_defense_next)))),
    L2_r = length(levels(
      dat_transition[[1]]$position_region_defense
    )),
    L2_c = length(levels(
      dat_transition[[1]]$position_region_defense_next
    )),
    L3_r = length(levels(dat_transition[[1]]$region_defense)),
    L3_c = length(levels(dat_transition[[1]]$region_defense_next)),
    # Raw data
    S_next = matrix(0,max(c(nrow(dat_transition[[1]]),
                            nrow(dat_transition[[2]]))),length(my_teams)),
    S_orig = matrix(0,max(c(nrow(dat_transition[[1]]),
                            nrow(dat_transition[[2]]))),length(my_teams)),
    S_time = matrix(0,max(c(nrow(dat_transition[[1]]),
                            nrow(dat_transition[[2]]))),length(my_teams)),
    # Connecting hierarchies
    L1_to_L2_team_1 = L1_to_L2[[1]],
    L1_to_L2_team_2 = L1_to_L2[[2]],
    L2_to_L3 = L2_to_L3,
    # Final-stage (Hyperprior) fixed values
    omega_mean = rep(0, length(unique(
      dat_transition[[1]]$time_int
    ))),
    half_cauchy_scale = 2.5
  )
  
  for(t in 1:2){
    transition_stan_input$S_next[1:nrow(dat_transition[[t]]), t] = as.numeric(dat_transition[[t]]$player_region_defense_next)
    transition_stan_input$S_orig[1:nrow(dat_transition[[t]]), t] = as.numeric(dat_transition[[t]]$player_region_defense)
    transition_stan_input$S_time[1:nrow(dat_transition[[t]]), t] = dat_transition[[t]]$time_int
  }

  return(list(stan_input = transition_stan_input,
              L1_dimnames = L1_levels,
              L2_dimnames = L2_levels,
              L3_dimnames = L3_levels))
}
