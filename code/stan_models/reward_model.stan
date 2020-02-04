data {
    // Defining dimensions
    int<lower=0> N_obs; // total observations
    int<lower=2> P; // number of players
    int<lower=2> H; // number of groups (new groups)
    int<lower=2> R; // number of court regions
    // Raw data
    int<lower=0,upper=1> M[N_obs]; // indexing make or miss
    int<lower=1,upper=P> M_player[N_obs]; // indexing player
  	int<lower=1,upper=H> M_group[N_obs]; // indexing player groups
  	int<lower=1,upper=R> M_region[N_obs]; // indexing court regions
  	int<lower=0,upper=1> M_open[N_obs];
    // Indexing players
  	int<lower=1,upper=H> player_group[P]; // indexing each player into a group
}

parameters {
  // Means
  matrix[P, R] mu; // player-specific court region effects
  matrix[H, R] psi; // group court region effects
	vector[R] varphi; // court region effects
  vector<lower = 0>[R] xi; //contested effects by court region
  // Variance parameters
	real<lower=0> sigma_mu; // governs court region effect uncertainty
	real<lower=0> sigma_psi; // governs group effect uncertainty
	real<lower=0> sigma_varphi; // global player effect uncertainty
	real<lower=0> sigma_xi; // contested effect uncertainty
} 

model {
  // Likelihood
  for(i in 1:N_obs){
     M[i] ~ bernoulli_logit(mu[M_player[i], M_region[i]] + M_open[i]*xi[M_region[i]]);
  }
  // First stage of hierarchy  
  for(p in 1:P){
		for(r in 1:R){
				mu[p, r] ~ normal(psi[player_group[p], r], sigma_mu);
		}
  }
  // Second stage of hierarchy -----> I'm here
  for(h in 1:H){
		for(r in 1:R){
				psi[h, r] ~ normal(varphi[r], sigma_psi);
		}
  }
  // Final-stage prior
  for(r in 1:R){
    varphi[r] ~ normal(0, sigma_varphi);
  }
  // Prior for xi
  for(r in 1:R){
    xi[r] ~ normal(0, sigma_xi);
  }
  //Variance hyperpriors
  sigma_mu ~ cauchy(0, 2.5);
  sigma_psi ~ cauchy(0, 2.5);
  sigma_varphi ~ cauchy(0, 2.5);
  sigma_xi ~ cauchy(0, 2.5);
}
generated quantities {
  matrix[P, R] mu_trans_cont;
  matrix[P, R] mu_trans_open;
  for(p in 1:P){
		for(r in 1:R){
				mu_trans_cont[p, r] = exp(mu[p, r])/(1 + exp(mu[p, r]));
				mu_trans_open[p, r] = exp(mu[p, r] + xi[r])/(1 + exp(mu[p, r] + xi[r]));
		}
  }
}