data {
    // Defining dimensions
    int<lower=0> N_obs; // number of observations
    int<lower=2> L1; // dimension of level 1 (players by court region)
    int<lower=2> L2; // dimension of level 2 (positions by court region)
    int<lower=2> L3; // base model row space (court region)
    // Raw data
    int<lower=0,upper=1> M[N_obs]; // indexing make or miss
    int<lower=1,upper=L1> M_player_region[N_obs]; // indexing player_region
  	int<lower=1,upper=L3> M_region[N_obs]; // indexing court regions
  	int<lower=0,upper=1> M_open[N_obs]; // indexing open/contested 
    // Connecting hierarchy indexes
    int<lower=1> L1_to_L2[L1]; // index of player positions
    int<lower=1> L2_to_L3[L2]; // index of player positions
  	// Final-stage (Hyperprior) fixed values
  	real varphi_mean; // hyperprior varphi mean
  	real half_cauchy_scale; //  hyperprior scale value of SDs
}

parameters {
  // Means
  vector[L1] mu; // player-specific court region effects
  vector[L2] psi; // group-specific court region effects
	vector[L3] varphi; // global court region effects
  vector<lower = 0>[L3] xi; //contested effects by court region
  // Variance parameters
	real<lower=0> sigma_mu; // L1 effect uncertainty
	real<lower=0> sigma_psi; // L2 effect uncertainty
	real<lower=0> sigma_varphi; // L3 effect uncertainty
	real<lower=0> sigma_xi; // contested effect uncertainty
} 

model {
  // Likelihood
  for(i in 1:N_obs){
     M[i] ~ bernoulli_logit(mu[M_player_region[i]] + M_open[i]*xi[M_region[i]]);
  }
  // First stage of hierarchy  
  for(l in 1:L1){
		 mu[l] ~ normal(psi[L1_to_L2[l]], sigma_mu);
  }
  // Second stage of hierarchy
  for(l in 1:L2){
		 psi[l] ~ normal(varphi[L2_to_L3[l]], sigma_psi);
  }
  // Final-stage prior
  for(l in 1:L3){
    varphi[l] ~ normal(varphi_mean, sigma_varphi);
  }
  // Prior for xi
  for(l in 1:L3){
    xi[l] ~ normal(0, sigma_xi);
  }
  //Variance hyperpriors
  sigma_mu ~ cauchy(0, half_cauchy_scale);
  sigma_psi ~ cauchy(0, half_cauchy_scale);
  sigma_varphi ~ cauchy(0, half_cauchy_scale);
  sigma_xi ~ cauchy(0, half_cauchy_scale);
}
