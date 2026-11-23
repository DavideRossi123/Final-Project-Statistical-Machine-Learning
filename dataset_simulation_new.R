# SIMULATION FUNCTION ----
# ordinal_data(): generates train/test data with an ordinal response y (1 < ... < K)
#
# ARGUMENTS
#   n_train       number of rows in the training set
#   n_test        number of rows in the test set (0 = no test set)
#   K             number of ordered categories
#   p0            how balanced the categories are: a vector of K probabilities,
#                 or "equi" (all equal), "middle" (middle categories frequent),
#                 "extreme" (extreme categories frequent), "skewed" (top categories rare)
#   p             number of covariates that really affect y (V1..Vp)
#   p_noise       number of irrelevant covariates (N1..), useful to test lasso
#   beta          relative weights of the p covariates (zeros allowed);
#                 the real coefficients are returned as beta_true
#   signal        strength of the effects, between 0 and 1
#                 (small = mostly noise, large = easy to predict)
#   relationship  shape of the effect: "linear", "quadratic", "interaction", "smooth"
#   rel_strength  how much of the effect is nonlinear, between 0 and 1
#                 (ignored if relationship = "linear")
#   np_strength   violation of proportional odds: 0 = none, 1 = strong
#   thresholds    leave NULL or pass a previous result's $thresholds 
#   n_ref         size of the sample used for that calibration (leave as is)

ordinal_data <- function(n_train = 1000, n_test = 0,
                         K = 5,
                         p = 2, p_noise = 0,
                         beta = rep(1, p),
                         relationship = c("linear", "quadratic", "interaction", "smooth"),
                         rel_strength = 0.5,
                         signal = 0.3,
                         p0 = "equi",
                         np_strength = 0,
                         thresholds = NULL,
                         n_ref = 1e5) {
  
  # 1. CHECKS 
  relationship <- match.arg(relationship)
  if (K < 2) stop("K must be >= 2.")
  if (length(beta) != p || all(beta == 0))
    stop("beta needs length p and at least one non-zero entry.")
  if (relationship == "interaction" && p < 2) stop("Interaction requires p >= 2.")
  if (np_strength > 0 && K < 3) stop("Non-proportional odds needs K >= 3.")
  if (signal <= 0 || signal >= 1) stop("signal (latent R2) must be in (0, 1).")
  if (rel_strength < 0 || rel_strength > 1) stop("rel_strength must be in [0, 1].")
  
  # 2. CLASS BALANCE (target marginal probabilities p0)
  if (is.character(p0)) {
    type <- match.arg(p0, c("equi", "middle", "extreme", "skewed"))
    m  <- dnorm(seq(-1.5, 1.5, length.out = K))
    w0 <- switch(type,
                 equi    = rep(1, K),
                 middle  = m,                       # middle categories frequent
                 extreme = max(m) + min(m) - m,     # extreme categories frequent
                 skewed  = 0.7^(seq_len(K) - 1))    # rare top categories
    p0 <- w0 / sum(w0)
  }
  if (length(p0) != K || any(p0 <= 0) || abs(sum(p0) - 1) > 1e-8)
    stop("p0 must contain K positive probabilities summing to 1.")
  
  # 3. LINEAR PREDICTOR eta WITH CONTROLLED VARIANCE 
  # IDEA: we fix the signal-to-noise ratio, so scenarios differ only in the factor we change
  w <- if (relationship == "linear") 0 else rel_strength # share of the signal that is nonlinear (0 if linear)
  V <- signal / (1 - signal) * pi^2 / 3 # target var(eta), so that signal = var(eta)/(var(eta)+pi^2/3)
  beta_true <- sqrt(V * (1 - w)) * beta / sqrt(sum(beta^2)) # rescale beta so the linear part has variance V*(1-w)
  
  if (relationship == "smooth") {                        # remove linear part of tanh
    tc <- integrate(function(z) z * tanh(2 * z) * dnorm(z), -Inf, Inf)$value
    tv <- integrate(function(z) tanh(2 * z)^2 * dnorm(z), -Inf, Inf)$value - tc^2
  }
  nl <- function(Xs) switch(relationship,                # mean 0, var 1, uncorrelated with X
                            linear      = 0,
                            quadratic   = (Xs[, 1]^2 - 1) / sqrt(2),
                            interaction = Xs[, 1] * Xs[, 2],
                            smooth      = (tanh(2 * Xs[, 1]) - tc * Xs[, 1]) / sqrt(tv))
  eta_fun <- function(Xs) as.vector(Xs %*% beta_true) + sqrt(V * w) * nl(Xs)
  
  # 4. PROPORTIONAL vs NON-PROPORTIONAL ODDS 
  # P(Y <= k | x) = plogis(theta_k - a_k * eta);  all a_k = 1  <=>  PO
  a <- 1 + np_strength * seq(-0.5, 0.5, length.out = K - 1)
  # np_strength=0 -> covariates have same effect everywhere
  # np_strength>1 -> the effect is weaker at the first cutpoints and stronger at the last ones, 
  #                   so it changes across cutpoints.
  
  # 5. THRESHOLDS CALIBRATED ON THE MARGINAL CLASS PROBABILITIES 
  # Goal: place the thresholds so the categories respect the target proportions
  if (is.null(thresholds)) {
    eta_ref <- eta_fun(matrix(rnorm(n_ref * p), n_ref, p))
    target  <- cumsum(p0)[-K]
    thresholds <- vapply(seq_len(K - 1), function(k)
      uniroot(function(th) mean(plogis(th - a[k] * eta_ref)) - target[k],
              interval = c(-50, 50), tol = 1e-8)$root, numeric(1))
  } # to be saved and reused for same scenarios 
  
  # 6. DRAW ONE DATASET 
  draw <- function(n) {
    Xs <- matrix(rnorm(n * p), n, p, dimnames = list(NULL, paste0("V", seq_len(p))))
    X  <- Xs
    if (p_noise > 0)                                     # irrelevant covariates
      X <- cbind(Xs, matrix(rnorm(n * p_noise), n, p_noise,
                            dimnames = list(NULL, paste0("N", seq_len(p_noise)))))
    
    eta <- eta_fun(Xs)
    cum <- plogis(matrix(thresholds, n, K - 1, byrow = TRUE) - outer(eta, a))
    # For every row i and cutpoint k, compute P(Y <= k) = plogis(threshold_k - a_k * eta_i)
    
    crossing <- rep(FALSE, n)                            # non-PO curves may cross
    if (np_strength > 0) {
      crossing <- rowSums(cum[, -1, drop = FALSE] < cum[, -(K - 1), drop = FALSE]) > 0
      for (j in 2:(K - 1)) cum[, j] <- pmax(cum[, j], cum[, j - 1])
    }
    
    y <- 1L + rowSums(runif(n) > cum)                    # inverse-CDF sampling
    
    d <- as.data.frame(X)
    d$y <- factor(y, levels = seq_len(K), ordered = TRUE)
    probs <- cbind(cum, 1) - cbind(0, cum)               # true class probabilities
    colnames(probs) <- seq_len(K)
    attr(d, "eta")           <- eta
    attr(d, "true_probs")    <- probs
    attr(d, "prop_crossing") <- mean(crossing)
    d
  }
  
  # 7. OUTPUT 
  list(train = draw(n_train),
       test  = if (n_test > 0) draw(n_test) else NULL,
       thresholds = thresholds, beta_true = beta_true, a = a, p0 = p0,
       signal = signal, V = V)
}

# SCENARIOs DEFINITION ----

# Baseline scenario equals this except for what is listed
# Baseline scenario equals this except for what is listed
base <- list(n_train = 1000, n_test = 1000, K = 5, p = 2, p_noise = 0,
             beta = c(0.8, 0.5), p0 = "equi", signal = 0.3,
             relationship = "linear", rel_strength = 0.5, np_strength = 0)

scenarios <- list(
  baseline      = base,
  
  # 1. proportional vs non-proportional odds
  nonPO_half    = modifyList(base, list(np_strength = 0.5)),
  nonPO_full    = modifyList(base, list(np_strength = 1)),
  
  # 2. sample size (only the training set changes)
  n_250         = modifyList(base, list(n_train = 250)),
  n_4000        = modifyList(base, list(n_train = 4000)),
  
  # 3. strength of the coefficients
  signal_weak   = modifyList(base, list(signal = 0.1)),
  signal_strong = modifyList(base, list(signal = 0.5)),
  
  # 4. class balance
  p0_middle     = modifyList(base, list(p0 = "middle")),
  p0_extreme    = modifyList(base, list(p0 = "extreme")),
  p0_skewed     = modifyList(base, list(p0 = "skewed")),
  
  # 5. number of categories
  K_3           = modifyList(base, list(K = 3)),
  K_8           = modifyList(base, list(K = 8)),
  
  # 6. type of relationship
  rel_quadratic = modifyList(base, list(relationship = "quadratic")),
  rel_interact  = modifyList(base, list(relationship = "interaction")),
  rel_smooth    = modifyList(base, list(relationship = "smooth")),
  
  # extras for regularization (irrelevant covariates)
  noise_10      = modifyList(base, list(p_noise = 10)),
  
  # a few combinations
  nonPO_n250    = modifyList(base, list(np_strength = 1, n_train = 250)),
  K8_skewed     = modifyList(base, list(K = 8, p0 = "skewed"))
)

# names(scenarios)
# length(scenarios)
# Dataset drawing ----
scen1 <- do.call(ordinal_data, scenarios$baseline)


