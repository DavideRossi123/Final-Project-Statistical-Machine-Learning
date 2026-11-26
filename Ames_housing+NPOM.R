# =====================================================================
# AMES HOUSING (2009 COHORT, K = 6) - ORDINAL CLASSIFICATION BENCHMARK
# =====================================================================
# Workflow:
#   1. Load & prepare Ames Yr.Sold == 2009 data (n = 648, K = 6 tiers, p = 19 cols).
#   2. Run quick EDA (tier distribution & predictor correlation matrix).
#   3. Define shared utilities: CV fold splitter, PAVA isotonic repair,
#      unregularized NPOM (All-Threshold logistic loss L_AT), Lasso NPOM solvers
#      (Formulation 1: Adjacent-Difference Fusion; Formulation 2: Full Telescoping),
#      ordinalNet CV tuner, and unified model scorer.
#   4. PART I — Full-Dataset Training (In-Sample Fit & Naive CV Tuning):
#        4A. Non-regularized & REML models fitted on all n = 648 observations.
#        4B. Regularized models fitted on all n = 648 observations with naive 10-fold CV
#            lambda selection (useful for full-sample interpretation, but includes tuning leakage).
#   5. PART II — Out-of-Sample Evaluation (Strict No-Data-Leakage CV):
#        5A. Non-regularized & REML models via standard 10-fold stratified CV.
#        5B. Regularized models via nested 10x5 stratified CV (lambda tuned strictly inside tr).
#   6. PART III — Confrontation Tables (Tables 1, 1B Overfitting Decomposition, 2, 3) & Figures 1-4.

set.seed(2000)

# Ensure user-installed CRAN libraries are visible under non-interactive Rscript on Windows
if (!requireNamespace("glmnet", quietly = TRUE)) {
  win_lib <- "C:/Users/david/AppData/Local/R/win-library/4.6"
  if (dir.exists(win_lib)) .libPaths(c(win_lib, .libPaths()))
}

library(AmesHousing)
library(ggplot2)
library(MASS)
library(Matrix)
library(glmnet)
library(mgcv)
library(nnet)
library(ordinalNet)
library(patchwork)

multinom <- nnet::multinom

theme_report <- theme_minimal(base_size = 11) +
  theme(
    plot.title       = element_text(face = "bold", size = 11.5),
    plot.subtitle    = element_text(color = "gray35", size = 9.5),
    panel.grid.minor = element_blank(),
    strip.text       = element_text(face = "bold", size = 10)
  )


# =====================================================================
# 1. DATA PREPARATION & STRATIFIED CV FOLDS
# =====================================================================
df_raw <- as.data.frame(ames_raw)
names(df_raw) <- make.names(names(df_raw))

# Focus on a single sale year (2009, n = 648) so the finite-sample variance
# penalty of 100-param Multinomial / unregularized NPOM vs 24-param POM is clear.
df_2009 <- df_raw[df_raw$Yr.Sold == 2009, ]
df_2009$Garage.Type[is.na(df_2009$Garage.Type)] <- "None"  # NA encodes "no garage" in Ames

req_cols <- c(
  "Overall.Qual", "Gr.Liv.Area", "Total.Bsmt.SF", "Year.Built", "Year.Remod.Add",
  "Garage.Area", "Fireplaces", "Kitchen.AbvGr", "Wood.Deck.SF", "Open.Porch.SF",
  "Lot.Area", "Full.Bath", "Half.Bath", "Bsmt.Full.Bath", "Bedroom.AbvGr",
  "Enclosed.Porch", "Central.Air", "Foundation", "Garage.Type"
)
df_sub <- na.omit(df_2009[, req_cols])  # 648 complete observations

# Symmetrically collapse extreme 3-level tails (1-3 -> "<=3", 8-10 -> ">=8")
qual_levels <- c("<=3", "4", "5", "6", "7", ">=8")
qual_merged <- ifelse(df_sub$Overall.Qual <= 3, "<=3",
                      ifelse(df_sub$Overall.Qual >= 8, ">=8", as.character(df_sub$Overall.Qual)))

# Merge rare categorical levels and scale continuous covariates for stable log-odds optimization
fnd_grp <- ifelse(df_sub$Foundation == "PConc", "PConc",
                  ifelse(df_sub$Foundation == "CBlock", "CBlock", "Other"))
gar_att <- ifelse(df_sub$Garage.Type %in% c("Attchd", "BuiltIn"), "Att_BI", "Det_None")

df_model <- data.frame(
  Overall.Qual   = factor(qual_merged, levels = qual_levels, ordered = TRUE),
  GrLiv_1000     = df_sub$Gr.Liv.Area / 1000,
  Bsmt_1000      = df_sub$Total.Bsmt.SF / 1000,
  Year_Decade    = (df_sub$Year.Built - 1970) / 10,
  Remod_Decade   = (df_sub$Year.Remod.Add - 1970) / 10,
  Garage_100     = df_sub$Garage.Area / 100,
  Fireplaces     = as.numeric(df_sub$Fireplaces),
  Kitchen_AbvGr  = as.numeric(df_sub$Kitchen.AbvGr),
  WoodDeck_100   = df_sub$Wood.Deck.SF / 100,
  OpenPorch_100  = df_sub$Open.Porch.SF / 100,
  Lot_10000      = df_sub$Lot.Area / 10000,
  Full_Bath      = as.numeric(df_sub$Full.Bath),
  Half_Bath      = as.numeric(df_sub$Half.Bath),
  Bsmt_Full_Bath = as.numeric(df_sub$Bsmt.Full.Bath),
  Bedroom_AbvGr  = as.numeric(df_sub$Bedroom.AbvGr),
  EnclPorch_100  = df_sub$Enclosed.Porch / 100,
  Central_Air    = factor(df_sub$Central.Air),
  Foundation_grp = factor(fnd_grp, levels = c("CBlock", "PConc", "Other")),
  Garage_Att     = factor(gar_att)
)

n_obs      <- nrow(df_model)
n_classes  <- nlevels(df_model$Overall.Qual)
pred_vars  <- names(df_model)[-1L]
num_vars   <- pred_vars[sapply(df_model[, pred_vars], is.numeric)]
X_all      <- model.matrix(~ . - Overall.Qual, data = df_model)[, -1L, drop = FALSE]
feat_names <- colnames(X_all)
y_all      <- df_model$Overall.Qual

cat("Ames 2009 dataset:", n_obs, "obs x", ncol(X_all), "design features,", n_classes, "tiers\n")
print(table(y_all))

# Stratified fold generator so rare extreme tiers ("<=3": 16 homes) stay balanced across folds
make_folds <- function(y_fac, K, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  fid <- integer(length(y_fac))
  for (lvl in levels(y_fac)) {
    idx <- which(y_fac == lvl)
    fid[idx] <- sample(rep(seq_len(K), length.out = length(idx)))
  }
  list(fold_id = fid, fold_list = split(seq_along(y_fac), fid))
}

K_folds   <- 10L
K_inner   <- 5L
outer_cv  <- make_folds(y_all, K_folds, seed = 42L)
fold_id   <- outer_cv$fold_id
fold_list <- outer_cv$fold_list


# =====================================================================
# 2. QUICK EXPLORATORY DATA ANALYSIS
# =====================================================================
df_qual <- as.data.frame(table(Tier = y_all))
df_qual$Pct   <- 100 * df_qual$Freq / sum(df_qual$Freq)
df_qual$Label <- sprintf("%d\n(%.1f%%)", df_qual$Freq, df_qual$Pct)

p_eda_bar <- ggplot(df_qual, aes(x = Tier, y = Freq)) +
  geom_col(fill = "steelblue4", width = 0.72) +
  geom_text(aes(label = Label), vjust = -0.25, size = 3.2, lineheight = 0.9) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "Merged Overall Quality (K = 6) Distribution",
       subtitle = "Ames 2009 cohort (n = 648)",
       x = "Overall.Qual (<=3 to >=8)", y = "Properties") +
  theme_report +
  theme(panel.grid.major.x = element_blank())
print(p_eda_bar)

cor_mat <- cor(df_model[, num_vars])
df_cor  <- expand.grid(Var1 = colnames(cor_mat), Var2 = rownames(cor_mat), KEEP.OUT.ATTRS = FALSE)
df_cor$r       <- as.vector(cor_mat)
df_cor$Var1    <- factor(df_cor$Var1, levels = colnames(cor_mat))
df_cor$Var2    <- factor(df_cor$Var2, levels = rev(rownames(cor_mat)))
df_cor$txt_col <- ifelse(abs(df_cor$r) > 0.52, "white", "gray15")

p_eda_cor <- ggplot(df_cor, aes(x = Var1, y = Var2, fill = r)) +
  geom_tile(color = "white", linewidth = 0.35) +
  geom_text(aes(label = sprintf("%.2f", r), color = txt_col), size = 2.6) +
  scale_color_identity() +
  scale_fill_gradient2(low = "#1B365D", mid = "white", high = "#B22222",
                       midpoint = 0, limits = c(-1, 1), name = "Corr") +
  coord_fixed() +
  labs(title = "Correlation Matrix: 15 Numerical Predictors (Ames 2009)", x = NULL, y = NULL) +
  theme_report +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8.5),
    axis.text.y = element_text(size = 8.5),
    panel.grid  = element_blank()
  )
print(p_eda_cor)


# =====================================================================
# 3. REUSABLE HELPERS (NPOM, PAVA, TUNING, SCORING & PLOTTING)
# =====================================================================

# Safe clipping + row renormalization so log(0) or negative raw probs never produce Inf/NaN
clip_probs <- function(P) {
  P <- pmax(P, 1e-15)
  sweep(P, 1L, rowSums(P), "/")
}

cumprobs_to_probs <- function(F_raw, level_names = NULL) {
  F_ext <- cbind(0, F_raw, 1)
  P <- F_ext[, -1L, drop = FALSE] - F_ext[, -ncol(F_ext), drop = FALSE]
  if (!is.null(level_names)) colnames(P) <- level_names
  P
}

# Row-wise PAVA projection onto 0 <= F_1 <= ... <= F_{K-1} <= 1 to fix crossed thresholds
isotonic_repair <- function(F_raw, level_names = NULL) {
  F_iso <- F_raw
  bad <- which(apply(F_raw, 1L, is.unsorted))
  for (i in bad) F_iso[i, ] <- as.numeric(isoreg(F_raw[i, ])$yf)
  F_iso <- pmin(pmax(F_iso, 0), 1)
  P_iso <- cumprobs_to_probs(F_iso, level_names)
  P_iso[P_iso < 0] <- 0
  P_iso <- sweep(P_iso, 1L, rowSums(P_iso), "/")
  P_iso
}

# Unregularized NPOM trained under Composite Marginal / All-Threshold Logistic Loss (L_AT)
fit_npom_lcum <- function(X, y_fac) {
  y_num <- as.integer(y_fac); K <- nlevels(y_fac); Km1 <- K - 1L
  X_glm <- cbind(1, -X)
  th <- numeric(Km1); B <- matrix(0, Km1, ncol(X))
  for (k in seq_len(Km1)) {
    fk <- suppressWarnings(glm.fit(X_glm, as.numeric(y_num <= k), family = binomial()))
    th[k] <- fk$coefficients[1L]
    B[k, ] <- fk$coefficients[-1L]
  }
  list(theta = th, beta_mat = B, levels = levels(y_fac), edf = Km1 * (ncol(X) + 1L))
}

predict_npom_lcum <- function(fit, newx, isotonic = TRUE) {
  eta   <- sweep(-as.matrix(newx) %*% t(fit$beta_mat), 2L, fit$theta, "+")
  F_raw <- plogis(eta)
  if (isotonic) isotonic_repair(F_raw, fit$levels) else cumprobs_to_probs(F_raw, fit$levels)
}

# Build stacked n*(K-1) binary design matrix for beta_k = v_0 + sum_{m=2}^k v_m
build_telescoping_stack <- function(X_std, Km1) {
  n <- nrow(X_std)
  I_th <- kronecker(Diagonal(Km1), rep(1, n))
  X_v0 <- kronecker(rep(1, Km1), -X_std)
  if (Km1 == 1L) return(cbind(I_th, X_v0))
  L_inc <- Matrix(0, nrow = Km1, ncol = Km1 - 1L)
  for (k in 2:Km1) L_inc[k, 1:(k - 1L)] <- 1
  cbind(I_th, X_v0, kronecker(L_inc, -X_std))
}

# Fit L1-penalized NPOM path under L_AT via glmnet:
#   - "adjacent_diff" (F1-Adj): penalizes beta_k - beta_{k-1} only; v_0 stays unpenalized
#   - "telescoping"   (F2-Tel): penalizes both baseline v_0 and increments v_2..v_{K-1}
fit_npom_lasso_path <- function(X, y_fac, reg_type = c("adjacent_diff", "telescoping"),
                                lam_grid = NULL, nLambda = 30L, lambdaMinRatio = 1e-3,
                                thresh = 1e-5, maxit = 1e4) {
  reg_type <- match.arg(reg_type)
  X <- as.matrix(X); p <- ncol(X)
  y_levels <- levels(y_fac); K <- length(y_levels); Km1 <- K - 1L
  y_num <- as.integer(y_fac)

  # Standardize predictors internally so the L1 penalty treats all features comparably
  x_mean <- colMeans(X)
  x_sd   <- apply(X, 2L, sd)
  x_sd[!is.finite(x_sd) | x_sd == 0] <- 1
  X_std   <- scale(X, center = x_mean, scale = x_sd)
  X_stack <- build_telescoping_stack(X_std, Km1)
  z_stack <- as.numeric(unlist(lapply(seq_len(Km1), function(k) as.numeric(y_num <= k))))

  n_unpen <- if (reg_type == "adjacent_diff") Km1 + p else Km1
  pf      <- c(rep(0, n_unpen), rep(1, ncol(X_stack) - n_unpen))

  # Exact KKT threshold where the first penalized coordinate enters from the unpenalized fit
  if (is.null(lam_grid)) {
    fit_0 <- suppressWarnings(glm.fit(as.matrix(X_stack[, seq_len(n_unpen), drop = FALSE]),
                                      z_stack, family = binomial()))
    g_pen <- as.vector(crossprod(X_stack[, -seq_len(n_unpen), drop = FALSE],
                                 z_stack - fit_0$fitted.values)) / length(z_stack)
    lmax  <- max(abs(g_pen)) * mean(pf)
    if (!is.finite(lmax) || lmax <= 0) lmax <- 0.05
    lam_grid <- exp(seq(log(lmax * 1.02), log(lmax * lambdaMinRatio), length.out = nLambda))
  }

  gfit <- glmnet(X_stack, z_stack, family = "binomial", alpha = 1,
                 penalty.factor = pf, intercept = FALSE, standardize = FALSE,
                 lambda = lam_grid, control = list(thresh = thresh, maxit = maxit))

  C_mat <- as.matrix(gfit$beta); n_lam <- ncol(C_mat)
  thresh_names <- paste0(y_levels[-K], "|", y_levels[-1L])
  feat_names   <- colnames(X)
  th_list <- vector("list", n_lam); b_list <- vector("list", n_lam)
  v0_list <- vector("list", n_lam); Vi_list <- vector("list", n_lam)
  edf     <- integer(n_lam)

  # Unpack parameters back to the original (unstandardized) covariate scale
  for (l in seq_len(n_lam)) {
    co   <- C_mat[, l]
    th_s <- co[seq_len(Km1)]
    v0_s <- co[(Km1 + 1L):(Km1 + p)]
    Vi_s <- matrix(co[-seq_len(Km1 + p)], nrow = Km1 - 1L, ncol = p, byrow = TRUE)

    B_s <- matrix(0, Km1, p); B_s[1L, ] <- v0_s
    for (k in 2:Km1) B_s[k, ] <- B_s[k - 1L, ] + Vi_s[k - 1L, ]

    B_orig  <- sweep(B_s, 2L, x_sd, "/")
    th_orig <- th_s + as.vector(B_orig %*% x_mean)
    v0_orig <- v0_s / x_sd
    Vi_orig <- sweep(Vi_s, 2L, x_sd, "/")

    names(th_orig)    <- thresh_names
    names(v0_orig)    <- feat_names
    dimnames(B_orig)  <- list(thresh_names, feat_names)
    dimnames(Vi_orig) <- list(paste0("jump_", thresh_names[-1L]), feat_names)

    th_list[[l]] <- th_orig
    b_list[[l]]  <- B_orig
    v0_list[[l]] <- v0_orig
    Vi_list[[l]] <- Vi_orig
    edf[l]       <- Km1 + sum(abs(v0_s) > 1e-10) + sum(abs(Vi_s) > 1e-10)
  }

  list(reg_type = reg_type, lambda = gfit$lambda, theta_list = th_list, beta_list = b_list,
       v0_list = v0_list, V_inc_list = Vi_list, edf = edf, levels = y_levels, K = K)
}

predict_npom_lasso <- function(path_fit, whichLambda, newx, isotonic = TRUE) {
  eta   <- sweep(-as.matrix(newx) %*% t(path_fit$beta_list[[whichLambda]]),
                 2L, path_fit$theta_list[[whichLambda]], "+")
  F_raw <- plogis(eta)
  if (isotonic) isotonic_repair(F_raw, path_fit$levels) else cumprobs_to_probs(F_raw, path_fit$levels)
}

# Tune NPOM Lasso along the full-data descending lambda grid via K-fold CV
tune_npom_lasso <- function(X, y_fac, folds, reg_type = c("adjacent_diff", "telescoping"),
                            nLambda = 30L, lambdaMinRatio = 1e-3) {
  reg_type <- match.arg(reg_type)
  full_fit <- fit_npom_lasso_path(X, y_fac, reg_type = reg_type,
                                  nLambda = nLambda, lambdaMinRatio = lambdaMinRatio)
  lam_grid <- full_fit$lambda; nl <- length(lam_grid); nf <- length(folds)
  cv_nll <- matrix(NA_real_, nl, nf)

  for (f in seq_len(nf)) {
    te <- folds[[f]]; tr <- setdiff(seq_len(nrow(X)), te)
    fp_f <- fit_npom_lasso_path(X[tr, , drop = FALSE], y_fac[tr], reg_type = reg_type, lam_grid = lam_grid)
    y_te <- as.integer(y_fac[te])
    for (l in seq_len(length(fp_f$lambda))) {
      P_c <- clip_probs(predict_npom_lasso(fp_f, l, X[te, , drop = FALSE], isotonic = TRUE))
      cv_nll[l, f] <- mean(-log(P_c[cbind(seq_along(y_te), y_te)]))
    }
  }

  mean_nll   <- rowMeans(cv_nll, na.rm = TRUE)
  pooled_nll <- as.vector(cv_nll %*% lengths(folds)) / sum(lengths(folds))
  se_nll     <- apply(cv_nll, 1L, sd, na.rm = TRUE) / sqrt(nf)
  best_idx   <- which.min(mean_nll)
  # lam_grid descends from strongest to weakest penalty -> min(which(...)) picks sparsest +1SE fit
  idx_1se    <- min(which(mean_nll <= mean_nll[best_idx] + se_nll[best_idx]))

  list(reg_type = reg_type, fit_full = full_fit, lam_grid = lam_grid, mean_nll = mean_nll,
       pooled_nll = pooled_nll, se_nll = se_nll, best_idx = best_idx, idx_1se = idx_1se,
       edf_min = full_fit$edf[best_idx], edf_1se = full_fit$edf[idx_1se],
       naive_nll_min = pooled_nll[best_idx], naive_nll_1se = pooled_nll[idx_1se])
}

extract_npom_preds <- function(tn_res, X_eval) {
  list(
    min_raw  = predict_npom_lasso(tn_res$fit_full, tn_res$best_idx, X_eval, isotonic = FALSE),
    min_iso  = predict_npom_lasso(tn_res$fit_full, tn_res$best_idx, X_eval, isotonic = TRUE),
    p1se_raw = predict_npom_lasso(tn_res$fit_full, tn_res$idx_1se,  X_eval, isotonic = FALSE),
    p1se_iso = predict_npom_lasso(tn_res$fit_full, tn_res$idx_1se,  X_eval, isotonic = TRUE)
  )
}

# Tune ordinalNet POM penalties (reverse = TRUE matches polr's sign convention)
tune_onet <- function(x_mat, y_fac, folds, alpha) {
  tn <- ordinalNetTune(
    x = x_mat, y = y_fac, folds = folds, alpha = alpha, standardize = TRUE,
    family = "cumulative", link = "logit", reverse = TRUE,
    parallelTerms = TRUE, nonparallelTerms = FALSE,
    nLambda = 30, lambdaMinRatio = 1e-4, threshIn = 1e-4, threshOut = 1e-4,
    printProgress = FALSE, warn = FALSE
  )
  cv_nll     <- sweep(-tn$loglik, 2L, lengths(folds), "/")
  mean_nll   <- rowMeans(cv_nll, na.rm = TRUE)
  pooled_nll <- as.numeric(rowSums(-tn$loglik) / sum(lengths(folds)))
  se_nll     <- apply(cv_nll, 1L, sd, na.rm = TRUE) / sqrt(length(folds))
  best_idx   <- which.min(mean_nll)
  idx_1se    <- min(which(mean_nll <= mean_nll[best_idx] + se_nll[best_idx]))
  list(fit_full = tn$fit, lam_grid = tn$lambdaVals, mean_nll = mean_nll, pooled_nll = pooled_nll,
       se_nll = se_nll, best_idx = best_idx, idx_1se = idx_1se,
       edf_min = as.numeric(tn$fit$nNonzero[best_idx]), edf_1se = as.numeric(tn$fit$nNonzero[idx_1se]),
       naive_nll_min = pooled_nll[best_idx], naive_nll_1se = pooled_nll[idx_1se])
}

plot_cv_curve <- function(res, title, col_line = "navy", col_se = "cornflowerblue") {
  df_p <- data.frame(
    log_lam  = log10(res$lam_grid),
    mean_nll = res$mean_nll,
    lo       = res$mean_nll - res$se_nll,
    hi       = res$mean_nll + res$se_nll
  )
  l_min  <- log10(res$lam_grid[res$best_idx])
  l_1se  <- log10(res$lam_grid[res$idx_1se])
  h_1se  <- res$mean_nll[res$best_idx] + res$se_nll[res$best_idx]
  vlines <- data.frame(
    x    = c(l_min, l_1se),
    rule = factor(c("lambda_min", "lambda_1SE"), levels = c("lambda_min", "lambda_1SE"))
  )

  ggplot(df_p, aes(x = log_lam, y = mean_nll)) +
    geom_errorbar(aes(ymin = lo, ymax = hi), color = col_se, width = 0.06, linewidth = 0.5) +
    geom_line(color = col_line, linewidth = 0.8) +
    geom_point(color = col_line, size = 1.8) +
    geom_hline(yintercept = h_1se, color = "gray55", linetype = "dotted", linewidth = 0.6) +
    geom_vline(data = vlines, aes(xintercept = x, color = rule, linetype = rule), linewidth = 0.8) +
    scale_color_manual(values = c("lambda_min" = "red3", "lambda_1SE" = "forestgreen"), name = NULL) +
    scale_linetype_manual(values = c("lambda_min" = "dashed", "lambda_1SE" = "dotdash"), name = NULL) +
    labs(title = title, x = expression(log[10](lambda)), y = "10-Fold CV NLL") +
    theme_report
}

# Unified scoring: all models are evaluated identically on held-out Exact Multinomial NLL, RPS, Accuracy & MAE
score_model <- function(name, y, prob_train, prob_oof, edf) {
  y_idx  <- as.integer(y)
  P_tr   <- clip_probs(prob_train)
  P_cv   <- clip_probs(prob_oof)
  nll_tr <- -log(P_tr[cbind(seq_len(n_obs), y_idx)])
  nll_cv <- -log(P_cv[cbind(seq_len(n_obs), y_idx)])
  ll_tr  <- -sum(nll_tr)
  y_hat  <- max.col(P_cv, ties.method = "first")
  f_nll  <- tapply(nll_cv, fold_id, mean)

  F_cv  <- t(apply(prob_oof, 1L, cumsum))[, seq_len(n_classes - 1L), drop = FALSE]
  O_mat <- outer(y_idx, seq_len(n_classes - 1L), "<=") * 1.0

  data.frame(
    Model        = name,
    EDF          = round(edf, 1),
    Train_LogLik = round(ll_tr, 2),
    Train_NLL    = round(mean(nll_tr), 4),
    CV_LogLik    = round(-sum(nll_cv), 2),
    CV_NLL       = round(mean(nll_cv), 4),
    SE_NLL       = round(sd(f_nll) / sqrt(K_folds), 4),
    Overfit_Gap  = round(mean(nll_cv) - mean(nll_tr), 4),
    RPS          = round(mean(rowSums((F_cv - O_mat)^2)), 4),
    AIC          = round(-2 * ll_tr + 2 * edf, 2),
    BIC          = round(-2 * ll_tr + log(n_obs) * edf, 2),
    Accuracy     = round(mean(y_hat == y_idx), 4),
    MAE_Class    = round(mean(abs(y_hat - y_idx)), 4),
    Neg_Probs    = sum(prob_oof < -1e-6, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}


# =====================================================================
# 4. PART I: FULL-DATASET TRAINING (IN-SAMPLE & NAIVE CV TUNING)
# =====================================================================
# Fits every model on all n = 648 observations to extract full-sample slopes,
# GAM smooth shapes, EDF/AIC/BIC, and in-sample training NLL.
# Note: for regularized models (4B), lambda is selected via 10-fold CV across
# the whole dataset, so its Naive_CV_NLL contains tuning leakage; Part II (5B)
# removes this leakage with nested 10x5 CV so both can be contrasted directly.

# ---------------------------------------------------------------------
# 4A. Non-Regularized & REML Models Fitted on Full Dataset (n = 648)
# ---------------------------------------------------------------------
cat("\n=== PART I (4A): FULL-DATA NON-REGULARIZED & REML FITS ===\n")

# Model 0: Unordered Multinomial Logit ((K - 1) * (p + 1) = 100 params)
df_nom <- df_model; df_nom$Overall.Qual <- factor(y_all, ordered = FALSE)
model_multinom <- multinom(Overall.Qual ~ ., data = df_nom, maxit = 1000, MaxNWts = 5000, trace = FALSE)
prob_multinom  <- predict(model_multinom, newdata = df_nom, type = "probs")

# Model 1: Cumulative Proportional Odds Logit (p + K - 1 = 24 params)
model_pom_logit <- polr(Overall.Qual ~ ., data = df_model, method = "logistic", Hess = TRUE)
prob_pom_logit  <- predict(model_pom_logit, newdata = df_model, type = "probs")
cat("\n=== MODEL 1: POM LOGIT SUMMARY ===\n")
print(summary(model_pom_logit))

# Model 2: Semi-Parametric Ordinal GAM (mgcv::ocat, REML splines on 5 core continuous covariates)
df_gam <- df_model; df_gam$Overall_Qual_int <- as.integer(y_all)
gam_formula <- Overall_Qual_int ~
  s(GrLiv_1000, k = 5) + s(Bsmt_1000, k = 5) + s(Year_Decade, k = 5) +
  s(Remod_Decade, k = 5) + s(Garage_100, k = 5) +
  Lot_10000 + Fireplaces + Kitchen_AbvGr + WoodDeck_100 + OpenPorch_100 +
  Full_Bath + Half_Bath + Bsmt_Full_Bath + Bedroom_AbvGr + EnclPorch_100 +
  Central_Air + Foundation_grp + Garage_Att

model_gam <- gam(gam_formula, family = ocat(R = n_classes), data = df_gam, method = "REML")
prob_gam  <- predict(model_gam, newdata = df_gam, type = "response")
cat("\n=== MODEL 2: ORDINAL GAM SUMMARY ===\n")
print(summary(model_gam))

# Plot partial smooth terms (+/- 2 SE) via ggplot2
gam_terms  <- predict(model_gam, newdata = df_gam, type = "terms", se.fit = TRUE)
gam_stable <- summary(model_gam)$s.table
smooth_vars <- c("GrLiv_1000", "Bsmt_1000", "Year_Decade", "Remod_Decade", "Garage_100")
df_smooth <- do.call(rbind, lapply(smooth_vars, function(v) {
  s_nm <- paste0("s(", v, ")")
  ord  <- order(df_gam[[v]])
  f_v  <- as.numeric(gam_terms$fit[ord, s_nm])
  se_v <- as.numeric(gam_terms$se.fit[ord, s_nm])
  data.frame(
    x          = df_gam[[v]][ord],
    fit        = f_v,
    lo         = f_v - 2 * se_v,
    hi         = f_v + 2 * se_v,
    term_label = sprintf("s(%s) [EDF = %.2f]", v, gam_stable[s_nm, "edf"]),
    stringsAsFactors = FALSE
  )
}))
df_smooth$term_label <- factor(df_smooth$term_label, levels = unique(df_smooth$term_label))
p_gam_smooth <- ggplot(df_smooth, aes(x = x, y = fit)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray55", linewidth = 0.5) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = "forestgreen", alpha = 0.22) +
  geom_line(color = "forestgreen", linewidth = 0.9) +
  facet_wrap(~ term_label, scales = "free_x", ncol = 3) +
  labs(title = "Model 2: Ordinal GAM Smooth Partial Effects (+/- 2 SE)",
       x = "Covariate Value", y = "Latent Scale Effect") +
  theme_report
print(p_gam_smooth)

# Model 4: Unregularized Cumulative NPOM (L_AT, Raw & PAVA Isotonic, 100 params)
model_npom_std    <- fit_npom_lcum(X_all, y_all)
prob_npom_std_raw <- predict_npom_lcum(model_npom_std, X_all, isotonic = FALSE)
prob_npom_std_iso <- predict_npom_lcum(model_npom_std, X_all, isotonic = TRUE)

# ---------------------------------------------------------------------
# 4B. Regularized Models Fitted on Full Dataset (Naive 10-Fold CV Lambda)
# ---------------------------------------------------------------------
cat("\n=== PART I (4B): FULL-DATA REGULARIZED FITS (NAIVE 10-FOLD CV TUNING) ===\n")

# Models 3a-3f: Penalized POM (ordinalNet Lasso, ElasticNet, Ridge)
penalties <- list(
  lasso = list(alpha = 1.0, label = "3a/b. ordinalNet Lasso"),
  enet  = list(alpha = 0.5, label = "3c/d. ordinalNet ENet (a=0.5)"),
  ridge = list(alpha = 0.0, label = "3e/f. ordinalNet Ridge (a=0)")
)
onet_results <- list()
onet_plots   <- list()

for (nm in names(penalties)) {
  cfg <- penalties[[nm]]
  res <- tune_onet(X_all, y_all, fold_list, cfg$alpha)
  res$prob_full_min <- predict(res$fit_full, newx = X_all, whichLambda = res$best_idx, type = "response")
  res$prob_full_1se <- predict(res$fit_full, newx = X_all, whichLambda = res$idx_1se,  type = "response")
  res$b_min         <- coef(res$fit_full, whichLambda = res$best_idx)[-(1:(n_classes - 1))][feat_names]
  res$b_1se         <- coef(res$fit_full, whichLambda = res$idx_1se)[-(1:(n_classes - 1))][feat_names]
  onet_results[[nm]] <- res

  cat(sprintf("%s: lam_min=%.5f (EDF=%d, Naive_CV=%.4f) | lam_1SE=%.5f (EDF=%d, Naive_CV=%.4f)\n",
              cfg$label, res$lam_grid[res$best_idx], res$edf_min, res$naive_nll_min,
              res$lam_grid[res$idx_1se], res$edf_1se, res$naive_nll_1se))
  onet_plots[[nm]] <- plot_cv_curve(res, cfg$label)
}
print(wrap_plots(onet_plots, ncol = 3, guides = "collect") & theme(legend.position = "bottom"))

# Models 5 & 6: Lasso-Regularized NPOM (Formulation 1: F1-Adj vs Formulation 2: F2-Tel)
npom_regs <- list(
  adj = list(type = "adjacent_diff", label = "5. NPOM Lasso F1-Adj (Unpen v0)", col = "darkorchid4", seg = "mediumpurple"),
  tel = list(type = "telescoping",   label = "6. NPOM Lasso F2-Tel (Pen v0)",   col = "darkcyan",    seg = "paleturquoise3")
)
npom_full_tune  <- list()
npom_full_probs <- list()
npom_plots      <- list()

for (rk in names(npom_regs)) {
  cfg <- npom_regs[[rk]]
  res <- tune_npom_lasso(X_all, y_all, fold_list, reg_type = cfg$type)
  npom_full_tune[[rk]]  <- res
  npom_full_probs[[rk]] <- extract_npom_preds(res, X_all)

  cat(sprintf("%s: lam_min=%.5f (EDF=%d, Naive_CV=%.4f) | lam_1SE=%.5f (EDF=%d, Naive_CV=%.4f)\n",
              cfg$label, res$lam_grid[res$best_idx], res$edf_min, res$naive_nll_min,
              res$lam_grid[res$idx_1se], res$edf_1se, res$naive_nll_1se))
  npom_plots[[rk]] <- plot_cv_curve(res, cfg$label, col_line = cfg$col, col_se = cfg$seg)
}
print(wrap_plots(npom_plots, ncol = 2, guides = "collect") & theme(legend.position = "bottom"))

# Inspect sparsity structure recovered on full Ames 2009 data at lambda_min
v0_tel_min <- npom_full_tune$tel$fit_full$v0_list[[npom_full_tune$tel$best_idx]]
dropped_v0 <- names(v0_tel_min)[abs(v0_tel_min) <= 1e-8]
cat("\nF2-Tel zeroed baseline covariates (v0 == 0 at lam_min):",
    if (length(dropped_v0) > 0L) paste(dropped_v0, collapse = ", ") else "None", "\n")

for (rk in names(npom_regs)) {
  tn_r  <- npom_full_tune[[rk]]
  V_min <- tn_r$fit_full$V_inc_list[[tn_r$best_idx]]
  nz    <- which(abs(V_min) > 1e-6, arr.ind = TRUE)
  cat(sprintf("Active non-parallel threshold jumps (%s, lam_min, EDF=%d): %d jumps\n",
              npom_regs[[rk]]$type, tn_r$edf_min, nrow(nz)))
}


# =====================================================================
# 5. PART II: OUT-OF-SAMPLE EVALUATION (STRICT NO-DATA-LEAKAGE CV)
# =====================================================================
# Every held-out observation in outer fold te is completely excluded from both
# parameter fitting and hyperparameter selection (no tuning or feature leakage).
cat("\n=== PART II (SECTION 5): OUT-OF-SAMPLE EVALUATION (NO DATA LEAKAGE) ===\n")

# ---------------------------------------------------------------------
# 5A. Non-Regularized & REML Models: Standard 10-Fold Stratified CV
# ---------------------------------------------------------------------
cat("Running 5A: Standard 10-fold CV for unpenalized & REML models...\n")
oof_multinom     <- matrix(NA_real_, n_obs, n_classes)
oof_pom_logit    <- matrix(NA_real_, n_obs, n_classes)
oof_gam          <- matrix(NA_real_, n_obs, n_classes)
oof_npom_std_raw <- matrix(NA_real_, n_obs, n_classes)
oof_npom_std_iso <- matrix(NA_real_, n_obs, n_classes)

for (k in seq_len(K_folds)) {
  te    <- fold_list[[k]]
  tr    <- setdiff(seq_len(n_obs), te)
  df_tr <- df_model[tr, ]; df_te <- df_model[te, ]

  df_tr_nom <- df_tr; df_tr_nom$Overall.Qual <- factor(df_tr$Overall.Qual, ordered = FALSE)
  df_te_nom <- df_te; df_te_nom$Overall.Qual <- factor(df_te$Overall.Qual, ordered = FALSE)
  fit_m <- multinom(Overall.Qual ~ ., data = df_tr_nom, maxit = 1000, MaxNWts = 5000, trace = FALSE)
  oof_multinom[te, ] <- predict(fit_m, newdata = df_te_nom, type = "probs")

  fit_pl <- polr(Overall.Qual ~ ., data = df_tr, method = "logistic", Hess = FALSE)
  oof_pom_logit[te, ] <- predict(fit_pl, newdata = df_te, type = "probs")

  df_tr_gam <- df_tr; df_tr_gam$Overall_Qual_int <- as.integer(df_tr$Overall.Qual)
  df_te_gam <- df_te; df_te_gam$Overall_Qual_int <- as.integer(df_te$Overall.Qual)
  fit_g <- gam(gam_formula, family = ocat(R = n_classes), data = df_tr_gam, method = "REML")
  oof_gam[te, ] <- predict(fit_g, newdata = df_te_gam, type = "response")

  fit_np_std <- fit_npom_lcum(X_all[tr, , drop = FALSE], y_all[tr])
  oof_npom_std_raw[te, ] <- predict_npom_lcum(fit_np_std, X_all[te, , drop = FALSE], isotonic = FALSE)
  oof_npom_std_iso[te, ] <- predict_npom_lcum(fit_np_std, X_all[te, , drop = FALSE], isotonic = TRUE)
}

# ---------------------------------------------------------------------
# 5B. Regularized Models: Nested 10x5 Stratified CV (Strict Out-of-Sample)
# ---------------------------------------------------------------------
# For every outer fold k = 1..10, tune lambda strictly inside tr (K_inner = 5)
# and predict untouched outer test fold te once -> eliminates Part I tuning leakage.
cat("Running 5B: Nested 10x5 CV for all regularized models...\n")
oof_lasso_min <- matrix(NA_real_, n_obs, n_classes)
oof_lasso_1se <- matrix(NA_real_, n_obs, n_classes)
oof_enet_min  <- matrix(NA_real_, n_obs, n_classes)
oof_enet_1se  <- matrix(NA_real_, n_obs, n_classes)
oof_ridge_min <- matrix(NA_real_, n_obs, n_classes)
oof_ridge_1se <- matrix(NA_real_, n_obs, n_classes)

oof_npom_las <- lapply(npom_regs, function(.) {
  list(
    min_raw  = matrix(NA_real_, n_obs, n_classes),
    min_iso  = matrix(NA_real_, n_obs, n_classes),
    p1se_raw = matrix(NA_real_, n_obs, n_classes),
    p1se_iso = matrix(NA_real_, n_obs, n_classes)
  )
})
names(oof_npom_las) <- names(npom_regs)

for (k in seq_len(K_folds)) {
  te          <- fold_list[[k]]
  tr          <- setdiff(seq_len(n_obs), te)
  X_tr        <- X_all[tr, , drop = FALSE]
  X_te        <- X_all[te, , drop = FALSE]
  y_tr        <- y_all[tr]
  inner_folds <- make_folds(y_tr, K_inner, seed = 100L + k)$fold_list

  tn_las_k <- tune_onet(X_tr, y_tr, inner_folds, alpha = 1.0)
  tn_ent_k <- tune_onet(X_tr, y_tr, inner_folds, alpha = 0.5)
  tn_rdg_k <- tune_onet(X_tr, y_tr, inner_folds, alpha = 0.0)

  oof_lasso_min[te, ] <- predict(tn_las_k$fit_full, newx = X_te, whichLambda = tn_las_k$best_idx, type = "response")
  oof_lasso_1se[te, ] <- predict(tn_las_k$fit_full, newx = X_te, whichLambda = tn_las_k$idx_1se,  type = "response")
  oof_enet_min[te, ]  <- predict(tn_ent_k$fit_full, newx = X_te, whichLambda = tn_ent_k$best_idx, type = "response")
  oof_enet_1se[te, ]  <- predict(tn_ent_k$fit_full, newx = X_te, whichLambda = tn_ent_k$idx_1se,  type = "response")
  oof_ridge_min[te, ] <- predict(tn_rdg_k$fit_full, newx = X_te, whichLambda = tn_rdg_k$best_idx, type = "response")
  oof_ridge_1se[te, ] <- predict(tn_rdg_k$fit_full, newx = X_te, whichLambda = tn_rdg_k$idx_1se,  type = "response")

  for (rk in names(npom_regs)) {
    tn_np_k <- tune_npom_lasso(X_tr, y_tr, inner_folds, reg_type = npom_regs[[rk]]$type)
    pr_k    <- extract_npom_preds(tn_np_k, X_te)
    for (slot_nm in names(pr_k)) oof_npom_las[[rk]][[slot_nm]][te, ] <- pr_k[[slot_nm]]
  }
}


# =====================================================================
# 6. PART III: CONFRONTATION TABLES, OVERFITTING CHECK & DIAGNOSTICS
# =====================================================================
edf_pom <- length(c(coef(model_pom_logit), model_pom_logit$zeta))
results_full <- rbind(
  score_model("0. Multinomial (Unordered)",        y_all, prob_multinom,                    oof_multinom,            model_multinom$edf),
  score_model("1. POM Logit (polr)",               y_all, prob_pom_logit,                   oof_pom_logit,           edf_pom),
  score_model("2. Ordinal GAM (mgcv ocat)",        y_all, prob_gam,                         oof_gam,                 attr(logLik(model_gam), "df")),
  score_model("3a. ordinalNet Lasso (min)",        y_all, onet_results$lasso$prob_full_min, oof_lasso_min,           onet_results$lasso$edf_min),
  score_model("3b. ordinalNet Lasso (1SE)",        y_all, onet_results$lasso$prob_full_1se, oof_lasso_1se,           onet_results$lasso$edf_1se),
  score_model("3c. ordinalNet ElasticNet (min)",   y_all, onet_results$enet$prob_full_min,  oof_enet_min,            onet_results$enet$edf_min),
  score_model("3d. ordinalNet ElasticNet (1SE)",   y_all, onet_results$enet$prob_full_1se,  oof_enet_1se,            onet_results$enet$edf_1se),
  score_model("3e. ordinalNet Ridge (min)",        y_all, onet_results$ridge$prob_full_min, oof_ridge_min,           onet_results$ridge$edf_min),
  score_model("3f. ordinalNet Ridge (1SE)",        y_all, onet_results$ridge$prob_full_1se, oof_ridge_1se,           onet_results$ridge$edf_1se),
  score_model("4a. NPOM Standard (Raw)",           y_all, prob_npom_std_raw,                oof_npom_std_raw,        model_npom_std$edf),
  score_model("4b. NPOM Standard (Isotonic)",      y_all, prob_npom_std_iso,                oof_npom_std_iso,        model_npom_std$edf),
  score_model("5a. NPOM Lasso F1-Adj (min, Raw)",  y_all, npom_full_probs$adj$min_raw,      oof_npom_las$adj$min_raw,  npom_full_tune$adj$edf_min),
  score_model("5b. NPOM Lasso F1-Adj (min, Iso)",  y_all, npom_full_probs$adj$min_iso,      oof_npom_las$adj$min_iso,  npom_full_tune$adj$edf_min),
  score_model("5c. NPOM Lasso F1-Adj (+1SE, Raw)", y_all, npom_full_probs$adj$p1se_raw,     oof_npom_las$adj$p1se_raw, npom_full_tune$adj$edf_1se),
  score_model("5d. NPOM Lasso F1-Adj (+1SE, Iso)", y_all, npom_full_probs$adj$p1se_iso,     oof_npom_las$adj$p1se_iso, npom_full_tune$adj$edf_1se),
  score_model("6a. NPOM Lasso F2-Tel (min, Raw)",  y_all, npom_full_probs$tel$min_raw,      oof_npom_las$tel$min_raw,  npom_full_tune$tel$edf_min),
  score_model("6b. NPOM Lasso F2-Tel (min, Iso)",  y_all, npom_full_probs$tel$min_iso,      oof_npom_las$tel$min_iso,  npom_full_tune$tel$edf_min),
  score_model("6c. NPOM Lasso F2-Tel (+1SE, Raw)", y_all, npom_full_probs$tel$p1se_raw,     oof_npom_las$tel$p1se_raw, npom_full_tune$tel$edf_1se),
  score_model("6d. NPOM Lasso F2-Tel (+1SE, Iso)", y_all, npom_full_probs$tel$p1se_iso,     oof_npom_las$tel$p1se_iso, npom_full_tune$tel$edf_1se)
)
results_full <- results_full[order(results_full$CV_NLL), ]

# --- TABLE 1: Master Model Comparison (Part I In-Sample vs Part II Leak-Free OOF) ---
cat("\n=== TABLE 1: MASTER MODEL COMPARISON (SORTED BY LEAK-FREE CV_NLL) ===\n")
print(results_full[, c("Model", "EDF", "Train_NLL", "CV_NLL", "SE_NLL",
                       "Overfit_Gap", "RPS", "AIC", "BIC", "Accuracy", "MAE_Class", "Neg_Probs")], row.names = FALSE)
cat("Negative OOF probability entries across all models:", sum(results_full$Neg_Probs), "\n")

# --- TABLE 1B: Overfitting & Tuning-Leakage Decomposition (Part I Naive CV vs Part II Nested CV) ---
naive_models <- c(
  "3a. ordinalNet Lasso (min)",
  "3b. ordinalNet Lasso (1SE)",
  "3c. ordinalNet ElasticNet (min)",
  "3d. ordinalNet ElasticNet (1SE)",
  "3e. ordinalNet Ridge (min)",
  "3f. ordinalNet Ridge (1SE)",
  "5b. NPOM Lasso F1-Adj (min, Iso)",
  "5d. NPOM Lasso F1-Adj (+1SE, Iso)",
  "6b. NPOM Lasso F2-Tel (min, Iso)",
  "6d. NPOM Lasso F2-Tel (+1SE, Iso)"
)
naive_vals <- as.numeric(c(
  onet_results$lasso$naive_nll_min,
  onet_results$lasso$naive_nll_1se,
  onet_results$enet$naive_nll_min,
  onet_results$enet$naive_nll_1se,
  onet_results$ridge$naive_nll_min,
  onet_results$ridge$naive_nll_1se,
  npom_full_tune$adj$naive_nll_min,
  npom_full_tune$adj$naive_nll_1se,
  npom_full_tune$tel$naive_nll_min,
  npom_full_tune$tel$naive_nll_1se
))
reg_sub <- results_full[match(naive_models, results_full$Model), ]
table_overfit <- data.frame(
  Model          = reg_sub$Model,
  EDF            = reg_sub$EDF,
  Train_NLL      = reg_sub$Train_NLL,
  Naive_CV_NLL   = round(naive_vals, 4),
  Nested_CV_NLL  = reg_sub$CV_NLL,
  Param_Overfit  = round(naive_vals - reg_sub$Train_NLL, 4),
  Tuning_Leakage = round(reg_sub$CV_NLL - naive_vals, 4),
  Total_Overfit  = reg_sub$Overfit_Gap,
  stringsAsFactors = FALSE
)
table_overfit <- table_overfit[order(table_overfit$Nested_CV_NLL), ]
cat("\n=== TABLE 1B: OVERFITTING & HYPERPARAMETER-LEAKAGE DECOMPOSITION (PART I vs PART II) ===\n")
print(table_overfit, row.names = FALSE)

# --- TABLE 2: Fitted Logit Slopes (min vs +1SE) & GAM Smooth EDF ---
gam_smry  <- summary(model_gam)
gam_edf_v <- sapply(feat_names, function(v) {
  sn <- paste0("s(", v, ")")
  if (sn %in% rownames(gam_smry$s.table)) gam_smry$s.table[sn, "edf"] else 1.0
})
cat("\n=== TABLE 2: LOGIT SLOPES, SHRINKAGE (MIN vs +1SE) & GAM EDF ===\n")
print(data.frame(
  Predictor = feat_names,
  POM_Logit = round(coef(model_pom_logit)[feat_names], 3),
  Lasso_min = round(onet_results$lasso$b_min, 3), Lasso_1SE = round(onet_results$lasso$b_1se, 3),
  ENet_min  = round(onet_results$enet$b_min,  3), ENet_1SE  = round(onet_results$enet$b_1se,  3),
  Ridge_min = round(onet_results$ridge$b_min, 3), Ridge_1SE = round(onet_results$ridge$b_1se, 3),
  GAM_EDF   = round(gam_edf_v, 2),
  stringsAsFactors = FALSE
), row.names = FALSE)

# --- TABLE 3: Per-Class Out-of-Fold Recall ---
oof_list <- list(
  Multinom  = oof_multinom,  POM_Logit = oof_pom_logit, Ord_GAM   = oof_gam,
  Lasso_min = oof_lasso_min, Lasso_1SE = oof_lasso_1se,
  ENet_min  = oof_enet_min,  ENet_1SE  = oof_enet_1se,
  Ridge_min = oof_ridge_min, Ridge_1SE = oof_ridge_1se,
  NPOM_Std  = oof_npom_std_iso,
  F1_Adj_m  = oof_npom_las$adj$min_iso,  F1_Adj_1s = oof_npom_las$adj$p1se_iso,
  F2_Tel_m  = oof_npom_las$tel$min_iso,  F2_Tel_1s = oof_npom_las$tel$p1se_iso
)
y_true_idx   <- as.integer(y_all)
table_recall <- data.frame(Overall_Qual = qual_levels, Count = as.integer(table(y_all)), stringsAsFactors = FALSE)
for (m_lbl in names(oof_list)) {
  y_pred <- max.col(clip_probs(oof_list[[m_lbl]]), ties.method = "first")
  table_recall[[m_lbl]] <- round(sapply(seq_len(n_classes), function(c) mean(y_pred[y_true_idx == c] == c)), 3)
}
cat("\n=== TABLE 3: PER-CLASS OUT-OF-FOLD RECALL ===\n")
print(table_recall, row.names = FALSE)


# --- FIGURES 1 & 2: Cross-Model Performance Bar Charts (ggplot2 + patchwork) ---
df_plot <- results_full
df_plot$Short_Name <- factor(gsub("ordinalNet ", "", df_plot$Model),
                             levels = rev(gsub("ordinalNet ", "", df_plot$Model)))
df_plot$Family <- factor(ifelse(
  grepl("Ordinal GAM", df_plot$Model), "Ordinal GAM",
  ifelse(grepl("Multinomial", df_plot$Model), "Multinomial",
         ifelse(grepl("F2-Tel", df_plot$Model), "NPOM Lasso F2-Tel",
                ifelse(grepl("NPOM", df_plot$Model), "NPOM Std / F1-Adj", "POM / ordinalNet")))
), levels = c("Ordinal GAM", "POM / ordinalNet", "NPOM Lasso F2-Tel", "NPOM Std / F1-Adj", "Multinomial"))

fam_palette <- c(
  "Ordinal GAM"       = "forestgreen",
  "POM / ordinalNet"  = "royalblue3",
  "NPOM Lasso F2-Tel" = "darkcyan",
  "NPOM Std / F1-Adj" = "darkorchid4",
  "Multinomial"       = "firebrick"
)

plot_hbar_gg <- function(df, metric_col, title, xlab, fmt = "%.3f", scale_pct = 1) {
  d <- data.frame(
    Short_Name = df$Short_Name,
    Family     = df$Family,
    val        = df[[metric_col]] * scale_pct
  )
  d$Label <- sprintf(fmt, d$val)
  ggplot(d, aes(x = val, y = Short_Name, fill = Family)) +
    geom_col(width = 0.72) +
    geom_text(aes(label = Label), hjust = -0.12, size = 3.0) +
    scale_fill_manual(values = fam_palette, name = "Model Family") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.16))) +
    labs(title = title, x = xlab, y = NULL) +
    theme_report +
    theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 8.5))
}

fig1 <- (plot_hbar_gg(df_plot, "CV_NLL",      "Fig 1A: Leak-Free CV NLL (Lower = Better)",      "Out-of-Fold NLL",    "%.3f") /
         plot_hbar_gg(df_plot, "Overfit_Gap", "Fig 1B: Total Overfit Gap (Lower = Better)",    "CV_NLL - Train_NLL", "%.3f")) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
print(fig1)

fig2 <- (plot_hbar_gg(df_plot, "Accuracy",  "Fig 2A: Classification Accuracy (Higher = Better)", "Accuracy (%)",      "%.1f%%", scale_pct = 100) /
         plot_hbar_gg(df_plot, "MAE_Class", "Fig 2B: Tier MAE (Lower = Better)",               "MAE_Class (Tiers)", "%.3f")) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
print(fig2)

# --- FIGURE 3: Standardized Feature Effects (POM Logit vs Lasso min & +1SE) ---
sd_x    <- apply(X_all, 2L, sd)
df_std  <- rbind(
  data.frame(Predictor = feat_names, Method = "POM Logit",   Effect = coef(model_pom_logit)[feat_names] * sd_x),
  data.frame(Predictor = feat_names, Method = "Lasso (min)", Effect = onet_results$lasso$b_min * sd_x),
  data.frame(Predictor = feat_names, Method = "Lasso (1SE)", Effect = onet_results$lasso$b_1se * sd_x)
)
df_std$Predictor <- factor(df_std$Predictor, levels = rev(feat_names))
df_std$Method    <- factor(df_std$Method, levels = c("POM Logit", "Lasso (min)", "Lasso (1SE)"))

fig3 <- ggplot(df_std, aes(x = Effect, y = Predictor, fill = Method)) +
  geom_vline(xintercept = 0, color = "gray45", linetype = "dashed", linewidth = 0.6) +
  geom_col(position = position_dodge(width = 0.78), width = 0.70) +
  scale_fill_manual(values = c("POM Logit" = "royalblue3", "Lasso (min)" = "darkorange2", "Lasso (1SE)" = "forestgreen"),
                    name = NULL) +
  labs(title = "Fig 3: Standardized Effects & Lasso +1SE Variable Selection",
       x = "Effect per 1-SD Increase in Predictor (Log-Odds Scale)", y = NULL) +
  theme_report +
  theme(panel.grid.major.y = element_blank(), legend.position = "top")
print(fig3)

# --- FIGURE 4: Out-of-Fold Confusion Matrices (Crisp Vector Tiles, ALL Percentages Shown) ---
cm_models <- list(
  "Fig 4A: 0. Multinomial"                = oof_multinom,
  "Fig 4B: 1. POM Logit"                  = oof_pom_logit,
  "Fig 4C: 6d. NPOM Lasso F2-Tel (+1SE)"  = oof_npom_las$tel$p1se_iso,
  "Fig 4D: 2. Ordinal GAM"                = oof_gam
)

df_cm <- do.call(rbind, lapply(names(cm_models), function(m_title) {
  y_hat <- factor(qual_levels[max.col(clip_probs(cm_models[[m_title]]), ties.method = "first")],
                  levels = qual_levels)
  cm    <- as.matrix(table(True = y_all, Pred = y_hat))
  cm_n  <- sweep(cm, 1L, pmax(rowSums(cm), 1), "/")
  d_m   <- expand.grid(True = qual_levels, Pred = qual_levels, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  d_m$Pct   <- as.vector(cm_n)
  d_m$Count <- as.vector(cm)
  d_m$Model <- m_title
  d_m
}))

df_cm$Model   <- factor(df_cm$Model, levels = names(cm_models))
df_cm$Pred    <- factor(df_cm$Pred, levels = qual_levels)
df_cm$True    <- factor(df_cm$True, levels = rev(qual_levels))
df_cm$Label   <- ifelse(df_cm$Count == 0, "0%", sprintf("%.1f%%", 100 * df_cm$Pct))
df_cm$TxtCol  <- ifelse(df_cm$Pct > 0.42, "white", ifelse(df_cm$Count == 0, "gray55", "gray10"))

fig4 <- ggplot(df_cm, aes(x = Pred, y = True, fill = Pct)) +
  geom_tile(color = "gray78", linewidth = 0.35) +
  geom_text(aes(label = Label, color = TxtCol), size = 2.85, fontface = "plain") +
  scale_color_identity() +
  scale_fill_gradientn(
    colours = c("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"),
    limits  = c(0, 1),
    labels  = function(x) sprintf("%.0f%%", 100 * x),
    name    = "Row Share"
  ) +
  coord_fixed() +
  facet_wrap(~ Model, ncol = 2) +
  labs(title = "Fig 4: Row-Normalized Out-of-Fold Confusion Matrices (All Cell Percentages)",
       subtitle = "Rows sum to 100% within each true Overall.Qual tier; exact zeros shown explicitly",
       x = "Predicted Tier", y = "True Tier") +
  theme_report +
  theme(panel.grid = element_blank(), legend.position = "right")
print(fig4)
