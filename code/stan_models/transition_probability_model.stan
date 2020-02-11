data {
  // Defining dimensions
  int<lower=1> N_obs; // number of observations
  int<lower=1> T_int; // number of time intervals
  int<lower=1> L1_r; // Level 1 row space of TPT
  int<lower=1> L1_c; // Level 1 column space of TPT
  int<lower=2> L2_r; // Level 2 row space of TPT
  int<lower=2> L2_c; // Level 2 column space of TPT
  int<lower=2> L3_r; // Level 3 row space of TPT
  int<lower=2> L3_c; // Level 3 column space of TPT
  // Raw data
  int<lower=1, upper=L1_c> S_next[N_obs]; 
  int<lower=1, upper=L1_r> S_orig[N_obs]; 
  int<lower=1, upper=T_int> S_time[N_obs];
  // Indexing players to positions
  int<lower=1> L1_to_L2[L1_c]; // index of player positions
  int<lower=1> L2_to_L3[L2_c];
  // int<lower=1> player_position[P]; // index of player positions
  // Final-stage (Hyperprior) fixed values
   row_vector[T_int] omega_mean; // this is the prior gamma values
}

parameters {
  matrix[L1_c, T_int] lambda[L1_r];
  matrix[L2_c, T_int] zeta[L2_r];
  matrix[L3_c, T_int] omega[L3_r];
  real<lower=0> sigma_lambda;
  real<lower=0> sigma_zeta;
  real<lower=0> sigma_omega;
  real<lower=0,upper = 1> rho;
}

transformed parameters {
  matrix[T_int, T_int] Sigma_lambda;
  matrix[T_int, T_int] Sigma_zeta;
  matrix[T_int, T_int] Sigma_omega;
  for(i in 1:T_int){
    for(j in 1:T_int){
      Sigma_lambda[i,j] = pow(sigma_lambda, 2)*pow(rho, abs(i-j)); // define Sigma_lambda
      Sigma_zeta[i,j] = pow(sigma_zeta,2)*pow(rho, abs(i-j)); // define Sigma_zeta
      Sigma_omega[i,j] = pow(sigma_omega,2)*pow(rho, abs(i-j)); // define Sigma_omega
    }
  }
}

model {
  for(n in 1:N_obs){
    S_next[n] ~ categorical_logit(col(lambda[S_orig[n]], S_time[n]));
  }
  for(i in 1:L1_r){
    for(j in 1:L1_c){
      lambda[i][j, ] ~ multi_normal(zeta[L1_to_L2[i]][L1_to_L2[j]], Sigma_lambda);
    }
  }
  for(i in 1:L2_r){
    for(j in 1:L2_c){
      zeta[i][j, ] ~ multi_normal(omega[L2_to_L3[i]][L2_to_L3[j]], Sigma_zeta);
    }
  }
  for(i in 1:L3_r){
    for(j in 1:L3_c){
      omega[i][j, ] ~ multi_normal(omega_mean, Sigma_omega);
    }
  }
  // AR(1) Hyperpriors
  sigma_lambda ~ cauchy(0, 2.5);
  sigma_zeta ~ cauchy(0, 2.5);
  sigma_omega ~ cauchy(0, 2.5);
  rho ~ uniform(0, 1);
}

// generated quantities {
//   matrix[K_m,L_m] phat[N];
//   matrix[K_g,L_g] phatG[N];
//   for(n in 1:N){
//     for(k in 1:K_m){
//       phat[n][k] = to_row_vector(softmax(theta[k][,n]));
//     }
//     for(k in 1:K_g){
//       phatG[n][k] = to_row_vector(softmax(thetaG[k][,n]));
//     }
//   }
// }



