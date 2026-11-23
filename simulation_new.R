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
  K_6           = modifyList(base, list(K = 6)),
  K_10          = modifyList(base, list(K = 10)),
  
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

# Calibrating thresholds (optional)
set.seed(611)
for (nm in names(scenarios)) {
  scenarios[[nm]]$thresholds <- do.call(
    ordinal_data, modifyList(scenarios[[nm]], list(n_train = 10, n_test = 0)))$thresholds
}

# PERFORMANCE ASSESSMENT FUNCTION ---- 
# metrics_from_probs() 
#   - takes a matrix of predicted probabilities and the true lables
#   - returns four metrics of model fit assessment 
# INPUT
#   - P: nxK matrix of model's predicted probabilities of each category for obs
#   - y: true labels 

metrics_from_probs <- function(P, y) {
  y_idx <- as.integer(y) # turn the true categories into a number
  K <- ncol(P) # reads K from the matrix 
  P <- pmax(P, 1e-15) # replace 0 probs with a small number (safe for log)
  P <- P / rowSums(P) # rescale probs so that they sum to 1 
  
  y_hat <- max.col(P, ties.method = "first") # predicts the category as the one with highest prob
  
  cumP <- P 
  for (j in 2:K) cumP[, j] <- cumP[, j - 1] + P[, j] # predicted cumulative distribution P(Y ≤ k)
  
  cumY <- outer(y_idx, seq_len(K - 1), "<=") * 1
  c(MZE = mean(y_hat != y_idx),                                    # 1 - accuracy
    MAE = mean(abs(y_hat - y_idx)),                                # error in tiers
    NLL = -mean(log(P[cbind(seq_along(y_idx), y_idx)])),           # log-loss
    RPS = mean(rowSums((cumP[, -K, drop = FALSE] - cumY)^2)) / (K - 1))
}

# OUTPUT
# - MZE: Share of observations where the top category is wrong (1 − accuracy)	
# - MAE: Average distance, in categories, between predicted and true category	
# - NLL: Average of −log(probability given to the true category)	
# - RPS: Squared gap between predicted and true cumulative distributions, 
#         averaged over cutpoints (divided by K−1)

# MODELs ----

make_folds <- function(y_fac, K, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  fid <- integer(length(y_fac))
  for (lvl in levels(y_fac)) {
    idx <- which(y_fac == lvl)
    fid[idx] <- sample(rep(seq_len(K), length.out = length(idx)))
  }
  list(fold_id = fid, fold_list = split(seq_along(y_fac), fid))
}

tune_onet <- function(x_mat, y_fac, folds, alpha) {
  tn <- ordinalNetTune(
    x = x_mat, y = y_fac, folds = folds, alpha = alpha, standardize = TRUE,
    family = "cumulative", link = "logit", reverse = TRUE,
    parallelTerms = TRUE, nonparallelTerms = FALSE,
    nLambda = 30, lambdaMinRatio = 1e-4, threshIn = 1e-4, threshOut = 1e-4,
    printProgress = FALSE, warn = FALSE
  )
  cv_nll   <- sweep(-tn$loglik, 2, lengths(folds), "/")
  mean_nll <- rowMeans(cv_nll, na.rm = TRUE)
  se_nll   <- apply(cv_nll, 1, sd, na.rm = TRUE) / sqrt(length(folds))
  best_idx <- which.min(mean_nll)
  idx_1se  <- min(which(mean_nll <= mean_nll[best_idx] + se_nll[best_idx]))
  list(fit_full = tn$fit, lam_grid = tn$lambdaVals, mean_nll = mean_nll, se_nll = se_nll,
       best_idx = best_idx, idx_1se = idx_1se)
}

## Fitting functions ----
# install.packages("nnet")
# install.packages("mgcv")
# install.packages("ordinalNet")

library(MASS); library(nnet); library(mgcv); library(ordinalNet)

# Multinomial logit (unordered baseline)
m_multinom <- function(train, test) {
  tr <- train; tr$y <- factor(tr$y, ordered = FALSE)
  fit <- nnet::multinom(y ~ ., data = tr, maxit = 1000, MaxNWts = 5000, trace = FALSE)
  predict(fit, newdata = test, type = "probs")
}

# Proportional odds model; link = "logistic" or "probit" 
m_pom <- function(link) function(train, test) {
  fit <- MASS::polr(y ~ ., data = train, method = link, Hess = FALSE)
  predict(fit, newdata = test, type = "probs")
}

# Ordinal GAM (mgcv::ocat, REML): smooth effects on the signal covariates V*,
# linear effects on the noise covariates N* 
m_gam <- function(train, test, k = 5) {
  tr <- train; tr$y_int <- as.integer(tr$y)
  te <- test;  te$y_int <- as.integer(test$y)
  xs <- setdiff(names(train), "y") # every covariate name
  sm <- grep("^V", xs, value = TRUE) # signal covariates name 
  f  <- as.formula(paste("y_int ~", paste(c(sprintf("s(%s, k = %d)", sm, k),
                                            setdiff(xs, sm)), collapse = " + ")))
  fit <- mgcv::gam(f, family = mgcv::ocat(R = nlevels(train$y)), data = tr, method = "REML")
  predict(fit, newdata = te, type = "response")
}

# Penalized cumulative logit (ordinalNet). alpha: 1 = lasso, 0.5 = elastic net, 0 = ridge.
# rule: "min" or "1se". Lambda is chosen by 5-fold CV INSIDE the training set.
m_onet <- function(alpha, rule) function(train, test) {
  xs    <- setdiff(names(train), "y")
  X     <- as.matrix(train[, xs, drop = FALSE])
  folds <- make_folds(train$y, 5)$fold_list
  tn    <- tune_onet(X, train$y, folds, alpha)
  idx   <- if (rule == "min") tn$best_idx else tn$idx_1se
  predict(tn$fit_full, newx = as.matrix(test[, xs, drop = FALSE]),
          whichLambda = idx, type = "response")
}

models <- list(
  Multinom    = m_multinom,
  POM_logit   = m_pom("logistic"),
  POM_probit  = m_pom("probit"),
  GAM         = m_gam,
  Lasso_1se   = m_onet(1,   "1se"),
  ENet_1se    = m_onet(0.5, "1se"),
  Ridge_1se   = m_onet(0,   "1se")
)


###################################################################
# 1. BASELINE SCENARIO 
set.seed(611)

# Draw the baseline dataset 
o     <- do.call(ordinal_data, scenarios$baseline)
train <- o$train
test  <- o$test
K     <- nlevels(train$y)

# check: realized class balance vs target
round(rbind(target = o$p0,
            train  = as.numeric(prop.table(table(train$y))),
            test   = as.numeric(prop.table(table(test$y)))), 3)

# Model fitting 
both  <- rbind(train, test)
n_tr  <- nrow(train)
P_test  <- list()      # test probabilities, one matrix per model
rows    <- list()      # one row of metrics per model

# oracle: the TRUE probabilities = best achievable performance
P_true_tr <- attr(train, "true_probs")
P_true_te <- attr(test,  "true_probs")
P_test[["Oracle"]] <- P_true_te
rows[["Oracle"]] <- data.frame(Model = "Oracle",
                               t(metrics_from_probs(P_true_te, test$y)),
                               NLL_train = unname(metrics_from_probs(P_true_tr, train$y)["NLL"]))

# fitted models
for (nm in names(models)) {
  cat("fitting", nm, "...\n")
  P_all <- tryCatch(suppressWarnings(models[[nm]](train, both)),
                    error = function(e) { message("  FAILED: ", conditionMessage(e)); NULL })
  if (!(is.matrix(P_all) && ncol(P_all) == K && nrow(P_all) == nrow(both) && !anyNA(P_all))) next
  
  P_tr <- P_all[seq_len(n_tr), , drop = FALSE]
  P_te <- P_all[-seq_len(n_tr), , drop = FALSE]
  P_test[[nm]] <- P_te
  rows[[nm]] <- data.frame(Model = nm,
                           t(metrics_from_probs(P_te, test$y)),
                           NLL_train = unname(metrics_from_probs(P_tr, train$y)["NLL"]))
}

tab <- do.call(rbind, rows)
rownames(tab) <- NULL
tab$Overfit_gap <- tab$NLL - tab$NLL_train                       # as in his Overfit_Gap
for (m in c("MZE", "MAE", "NLL", "RPS"))                         # distance from the oracle
  tab[[paste0(m, "_gap")]] <- tab[[m]] - tab[[m]][tab$Model == "Oracle"]

print(tab[order(tab$NLL), ], digits = 3, row.names = FALSE)
