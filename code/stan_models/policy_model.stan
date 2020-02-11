data {
  // Defining dimensions
  int<lower=1> N_obs; // number of observations
  int<lower=1> T_int; // number of time intervals
  int<lower=2> L1; // dimension of (players by region by defense)
  int<lower=2> L2; // dimension of (positions by region by defense)
  int<lower=2> L3; // base model row space (region by defense)
  // Raw data
  int<lower=0, upper=1> A[N_obs]; 
  int<lower=1, upper=L1> A_state[N_obs]; 
  int<lower=1, upper=T_int> A_time[N_obs];
  // Connecting hierarchy indexes
  int<lower=1> L1_to_L2[L1]; 
  int<lower=1> L2_to_L3[L2];
  // Final-stage (Hyperprior) fixed values
  row_vector[T_int] gamma_mean; // this is the prior gamma values
}

parameters {
  // Means
  matrix[T_int, L1] theta;
  matrix[T_int, L2] beta;
  matrix[T_int, L3] gamma;
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
 for(n in 1:N_obs){
     A[n] ~ bernoulli_logit(theta[A_time[n], A_state[n]]);
 }
  // First stage of hierarchy  
  for(l in 1:L1){
    theta[ ,l] ~ multi_normal(beta[, L1_to_L2[l]], Sigma_theta);
  }
  // Second stage of hierarchy
  for(l in 1:L2){
    beta[, l] ~ multi_normal(gamma[, L2_to_L3[l]], Sigma_beta);
  }
  // Mean hyperpriors
  for(l in 1:L3){
    gamma[, l] ~ multi_normal(gamma_mean, Sigma_gamma);
  }
  // AR(1) Hyperpriors
  sigma_theta ~ cauchy(0, 2.5);
  sigma_beta ~ cauchy(0, 2.5);
  sigma_gamma ~ cauchy(0, 2.5);
  rho ~ uniform(0, 1);
}
generated quantities {
  matrix[T_int, L1] trans_theta;
  for(t in 1:T_int){
    for(l in 1:L1){
		trans_theta[t, l] = exp(theta[t, l])/(1 + exp(theta[t, l]));
    }
  }
}