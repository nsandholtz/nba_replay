data {
  // Defining dimensions
  int<lower=1> N_obs; // number of observations
  int<lower=1> T_int; // number of time intervals
  int<lower=2> P; // number of players
  int<lower=2> G; // number of positions
  int<lower=2> R; // number of court regions
  int<lower=2> K; // base model row space of TPT Slice (region by defense)
  // Raw data
  int<lower=0, upper=1> A[N_obs]; 
  int<lower=1, upper=K> A_states[N_obs]; 
  int<lower=1, upper=P> A_players[N_obs]; 
  int<lower=1, upper=R> A_regions[N_obs]; 
  int<lower=1, upper=T_int> A_time[N_obs];
  // Indexing players
  int<lower=1> player_position[P]; // index of player positions
  // Final-stage (Hyperprior) fixed values
  row_vector[T_int] gamma_mean; // this is the prior gamma values
}

parameters {
  // Means
  matrix[T_int, K] theta[P];
  matrix[T_int, K] beta[G];
  matrix[T_int, K] gamma;
  // AR(1) Parameters
  real<lower=0> sigma_theta;
  real<lower=0> sigma_beta;
  real<lower=0> sigma_gamma;
  real<lower=0, upper = 1> rho;
}

transformed parameters {
  // Define AR(1) matrices
  matrix[T_int, T_int] Sigma_theta;
  matrix[T_int,T_int] Sigma_beta;
  matrix[T_int,T_int] Sigma_gamma;
  for(i in 1:T_int){
    for(j in 1:T_int){
      Sigma_theta[i,j] = pow(sigma_theta, 2) * pow(rho, abs(i-j)); 
      Sigma_beta[i,j] = pow(sigma_beta, 2) * pow(rho, abs(i-j)); 
      Sigma_gamma[i,j] = pow(sigma_gamma, 2) * pow(rho, abs(i-j)); 
    }
  }
}

model {
 // Top level
 for(i in 1:N_obs){
   if (A_regions[i] != 4) // 4 denotes the 'heave' region
     A[i] ~ bernoulli_logit(theta[A_players[i]][A_time[i], A_states[i]]);
   else 
     A[i] ~ bernoulli_logit(gamma[A_time[i], A_states[i]]);
 }
  // First stage of hierarchy  
  for(p in 1:P){
    for(k in 1:K){
      theta[p][,k] ~ multi_normal(beta[player_position[p]][,k], Sigma_theta);
    }
  }
  // Second stage of hierarchy
  for(g in 1:G){
    for(k in 1:K){
      beta[g][,k] ~ multi_normal(gamma[,k], Sigma_beta);
    }
  }
  // Mean hyperpriors
  for(k in 1:K){
    gamma[,k] ~ multi_normal(gamma_mean, Sigma_gamma);
  }
  // AR(1) Hyperpriors
  sigma_theta ~ cauchy(0, 2.5);
  sigma_beta ~ cauchy(0, 2.5);
  sigma_gamma ~ cauchy(0, 2.5);
  rho ~ uniform(0, 1);
}


