# =====================================================================
# AMES HOUSING (2009 COHORT, K = 7) - ORDINAL CLASSIFICATION BENCHMARK
# =====================================================================
# Workflow:
#   1. Prepare Ames Yr.Sold == 2009 data (n = 648, K = 7 merged tiers, 18 covariates -> 19 design cols).
#   2. Exploratory checks (class counts & correlation matrix).
#   3. Fit all models on the full dataset (Multinomial, POM Logit/Probit, Ordinal GAM, ordinalNet).
#   4. Evaluate out-of-sample generalization cleanly in two separate blocks:
#        4A. Standard 10-fold stratified CV for non-regularized models (Multinomial, POM, GAM).
#        4B. Nested 10-fold CV (10 outer x 5 inner folds) for regularized ordinalNet models.
#   5. Print confrontation tables (Metrics 1, 2, 3) and diagnostic figures.

set.seed(2000)

library(AmesHousing)
library(MASS)
library(mgcv)
library(nnet)
library(ordinalNet)

multinom <- nnet::multinom

# Load raw Ames dataset and sanitize column names (spaces -> dots)
df_raw <- as.data.frame(ames_raw)
names(df_raw) <- make.names(names(df_raw))


# =====================================================================
# VARIABLE DICTIONARY
# =====================================================================
# Overall.Qual   : Ordered factor (7 tiers). The target variable representing the overall material and finish quality of the house.
# GrLiv_1000     : Numeric. Above-grade (ground level) living area, scaled in thousands of square feet.
# Bsmt_1000      : Numeric. Total basement area, scaled in thousands of square feet.
# Year_Decade    : Numeric. The year the house was originally built, scaled as decades relative to 1970 (e.g., 0 = 1970, 1 = 1980, -1 = 1960).
# Remod_Decade   : Numeric. The year of the last remodel or addition, scaled as decades relative to 1970.
# Garage_100     : Numeric. Size of the garage, scaled in hundreds of square feet.
# Fireplaces     : Numeric. Total number of fireplaces in the home.
# Kitchen_AbvGr  : Numeric. Number of kitchens located above grade.
# WoodDeck_100   : Numeric. Wood deck area, scaled in hundreds of square feet.
# OpenPorch_100  : Numeric. Open porch area, scaled in hundreds of square feet.
# Lot_10000      : Numeric. Total lot size/area, scaled in tens of thousands of square feet.
# Full_Bath      : Numeric. Number of full bathrooms located above grade.
# Half_Bath      : Numeric. Number of half bathrooms located above grade.
# Bsmt_Full_Bath : Numeric. Number of full bathrooms located in the basement.
# Bedroom_AbvGr  : Numeric. Number of bedrooms located above grade.
# EnclPorch_100  : Numeric. Enclosed porch area, scaled in hundreds of square feet.
# Central_Air    : Factor. Indicates the presence of central air conditioning.
# Foundation_grp : Factor. Broad structural grouping for the foundation type (Poured Concrete, Cinder Block, or Other).
# Garage_Att     : Factor. Broad structural grouping for the garage layout (Attached/Built-In vs. Detached/None).




# =====================================================================
#  Data cleaning and K-fold Data Preparation
# =====================================================================
# Focus on a single sale year (2009, n = 648) rather than pooling all years (~2,900 homes):
# in smaller finite samples, unordered Multinomial logit clearly suffers from fitting (K-1)*(p+1) params.
df_2009 <- df_raw[df_raw$Yr.Sold == 2009, ]
df_2009$Garage.Type[is.na(df_2009$Garage.Type)] <- "None"  # NA in Ames means no garage, not missing data

# Only 18 physical covariates were retained among the 81 potential regressors
# included in the model
req_cols <- c(
  "Overall.Qual", "Gr.Liv.Area", "Total.Bsmt.SF", "Year.Built", "Year.Remod.Add",
  "Garage.Area", "Fireplaces", "Kitchen.AbvGr", "Wood.Deck.SF", "Open.Porch.SF",
  "Lot.Area", "Full.Bath", "Half.Bath", "Bsmt.Full.Bath", "Bedroom.AbvGr",
  "Enclosed.Porch", "Central.Air", "Foundation", "Garage.Type")

df_sub <- na.omit(df_2009[, req_cols])  # 648 complete observations

# Collapse sparse extreme ratings (1,2,3 -> "<=3", 9,8,10 -> ">=9")
qual_levels <- c("<=3", "4", "5", "6", "7", "8", ">=9")
qual_merged <- ifelse(df_sub$Overall.Qual <= 3, "<=3",
               ifelse(df_sub$Overall.Qual >= 9, ">=9", as.character(df_sub$Overall.Qual)))

# Group rare categorical levels into broad structural buckets to prevent empty cells in CV
fnd_grp <- ifelse(df_sub$Foundation == "PConc", "PConc",ifelse(df_sub$Foundation == "CBlock", "CBlock", "Other"))
gar_att <- ifelse(df_sub$Garage.Type %in% c("Attchd", "BuiltIn"), "Att_BI", "Det_None")

# Build modeling data frame; divide large square-footage/year variables so slopes stay reasonably low
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
  Garage_Att     = factor(gar_att))

n_obs      <- nrow(df_model)
n_classes  <- nlevels(df_model$Overall.Qual)
pred_vars  <- names(df_model)[-1]
num_vars   <- pred_vars[sapply(df_model[, pred_vars], is.numeric)]

# Numeric design matrix with factors dummy-coded and intercept column dropped
X_all      <- model.matrix(~ . - Overall.Qual, data = df_model)[, -1, drop = FALSE]
feat_names <- colnames(X_all)
y_all      <- df_model$Overall.Qual

cat("Ames 2009 dataset:", n_obs, "obs x", ncol(X_all), "design features,", n_classes, "tiers\n")
print(table(y_all))

# Stratified K-fold splitter: shuffles indices within each response category so rare classes
# ("<=3" has only 16 homes) are evenly distributed across all K folds
make_folds <- function(y_fac, K, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  fid <- integer(length(y_fac))
  for (lvl in levels(y_fac)) {
    idx <- which(y_fac == lvl)
    fid[idx] <- sample(rep(seq_len(K), length.out = length(idx)))
  }
  list(fold_id = fid, fold_list = split(seq_along(y_fac), fid))
}

K_folds   <- 10  # outer evaluation folds (reused in 4A, 4B, and Section 3d path inspection)
K_inner   <- 5   # inner lambda-tuning folds strictly inside each training split in 4B
outer_cv  <- make_folds(y_all, K_folds, seed = 42)
fold_id   <- outer_cv$fold_id
fold_list <- outer_cv$fold_list


# =====================================================================
# QUICK EXPLORATORY DATA ANALYSIS
# =====================================================================
# Check balance across the 7 merged quality tiers before modeling
barplot(table(y_all), col = "steelblue", border = NA,
        main = "Merged Overall Quality (K = 7) Distribution (Ames 2009, n = 648)",
        xlab = "Overall.Qual (<=3 to >=9)", ylab = "Properties")

# Check collinearity among numeric covariates (reverse rows in image() so the 1.00 diagonal runs top-left -> bottom-right)
cor_mat <- cor(df_model[, num_vars])
par(mar = c(8.2, 8.2, 3.0, 1.8))
cor_pal <- colorRampPalette(c("navy", "white", "firebrick"))(100)
image(seq_len(ncol(cor_mat)), seq_len(nrow(cor_mat)), t(cor_mat[nrow(cor_mat):1, ]),
      col = cor_pal, zlim = c(-1, 1), axes = FALSE, xlab = "", ylab = "",
      main = "Correlation Matrix: 15 Numerical Predictors (Ames 2009)")
axis(1, at = seq_len(ncol(cor_mat)), labels = colnames(cor_mat), las = 2, cex.axis = 0.72)
axis(2, at = seq_len(nrow(cor_mat)), labels = rev(rownames(cor_mat)), las = 2, cex.axis = 0.72)
for (i in seq_len(nrow(cor_mat))) {
  for (j in seq_len(ncol(cor_mat))) {
    text(j, nrow(cor_mat) - i + 1, sprintf("%.2f", cor_mat[i, j]), cex = 0.56)
  }
}
par(mar = c(5.1, 4.1, 4.1, 2.1))


# =====================================================================
# FIT ALL MODELS ON FULL DATASET
# =====================================================================

# Model 0: Unordered Multinomial Logit ((K - 1) * (p + 1) = 120 params)
# Ignores tier ordering by fitting 6 separate log-odds equations
df_nom <- df_model
df_nom$Overall.Qual <- factor(y_all, ordered = FALSE)
model_multinom <- multinom(Overall.Qual ~ ., data = df_nom,
                           maxit = 1000, MaxNWts = 5000, trace = FALSE)
prob_multinom  <- predict(model_multinom, newdata = df_nom, type = "probs")

# Models 1a / 1b: Cumulative Proportional Odds Logit & Probit (p + K - 1 = 25 params)
# Exploits ordinal structure: P(Y <= k | x) = G(theta_k - w^T x), sharing one slope vector w
# across all 6 cutpoints. Hess = TRUE retains the Hessian for coefficient SEs;
# Probit checks link sensitivity.
model_pom_logit  <- polr(Overall.Qual ~ ., data = df_model, method = "logistic", Hess = TRUE)
model_pom_probit <- polr(Overall.Qual ~ ., data = df_model, method = "probit",   Hess = TRUE)
prob_pom_logit   <- predict(model_pom_logit,  newdata = df_model, type = "probs")
prob_pom_probit  <- predict(model_pom_probit, newdata = df_model, type = "probs")

cat("\n=== MODEL 1a: POM LOGIT SUMMARY ===\n")
print(summary(model_pom_logit))
print(summary(model_pom_probit))


# Model 2: Semi-Parametric Ordinal GAM (mgcv::ocat, REML smoothness)
# Keeps proportional odds on the latent scale while allowing smooth nonlinear effects
# (basis dimension capped at k = 5) only on the 5 core continuous size/vintage covariates.
# Non-numeric factors and discrete/secondary covariates stay linear.

df_gam <- df_model
df_gam$Overall_Qual_int <- as.integer(y_all)  # mgcv::ocat requires integer codes 1..K

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
plot(model_gam, pages = 1, shade = TRUE)

# Models 4a - 4d: Penalized Cumulative Logit
# Tunes lambda via K-fold CV across 30 descending grid values.
# reverse = TRUE matches polr()'s sign convention (w_j > 0 shifts probability
# toward higher quality).

# This implements a lambda selection across a 10 fold cross validation (overfitting over
# the training data to evalutate predictive performances a nested 10 fold CV will be
# necessary)
tune_onet <- function(x_mat, y_fac, folds, alpha) {
  tn <- ordinalNetTune(
    x = x_mat, y = y_fac, folds = folds, alpha = alpha, standardize = TRUE,
    family = "cumulative", link = "logit", reverse = TRUE,
    parallelTerms = TRUE, nonparallelTerms = FALSE,
    nLambda = 30, lambdaMinRatio = 1e-4, threshIn = 1e-4, threshOut = 1e-4,
    printProgress = FALSE, warn = FALSE
  )
  # Convert total fold log-likelihoods into per-observation NLL, then mean and SE across folds
  cv_nll   <- sweep(-tn$loglik, 2, lengths(folds), "/")
  mean_nll <- rowMeans(cv_nll, na.rm = TRUE)
  se_nll   <- apply(cv_nll, 1, sd, na.rm = TRUE) / sqrt(length(folds))
  best_idx <- which.min(mean_nll)
  # lambdaVals descends from strongest to weakest penalty, so min(which(...)) gives the sparsest +1 SE fit
  idx_1se  <- min(which(mean_nll <= mean_nll[best_idx] + se_nll[best_idx]))

  list(
    fit_full = tn$fit, lam_grid = tn$lambdaVals, mean_nll = mean_nll, se_nll = se_nll,
    best_idx = best_idx, idx_1se = idx_1se,
    edf_min  = tn$fit$nNonzero[best_idx], edf_1se = tn$fit$nNonzero[idx_1se]
  )
}

cat("\n=== SECTION 3d: PENALIZED ORDINAL LOGIT (FULL-DATA CV TUNING PATHS) ===\n")
penalties    <- list(
  lasso = list(alpha = 1.0, label = "4a/b. ordinalNet Lasso"),
  enet  = list(alpha = 0.5, label = "4c/d. ordinalNet ENet (a=0.5)"),
  ridge = list(alpha = 0.0, label = "4e/f. ordinalNet Ridge (a=0)")
)
onet_results <- list()

par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3.5, 1.5))
for (nm in names(penalties)) {
  cfg <- penalties[[nm]]
  res <- tune_onet(X_all, y_all, fold_list, cfg$alpha)

  # Store full-sample predictions and slopes (skipping K - 1 unpenalized cutpoints) for min and +1 SE
  res$prob_full_min <- predict(res$fit_full, newx = X_all, whichLambda = res$best_idx, type = "response")
  res$prob_full_1se <- predict(res$fit_full, newx = X_all, whichLambda = res$idx_1se,  type = "response")
  res$b_min         <- coef(res$fit_full, whichLambda = res$best_idx)[-(1:(n_classes - 1))][feat_names]
  res$b_1se         <- coef(res$fit_full, whichLambda = res$idx_1se)[-(1:(n_classes - 1))][feat_names]
  onet_results[[nm]] <- res

  cat(sprintf("%s: lambda_min = %.5f (EDF=%d, CV_NLL=%.4f) | lambda_1SE = %.5f (EDF=%d, CV_NLL=%.4f)\n",
              cfg$label, res$lam_grid[res$best_idx], res$edf_min, res$mean_nll[res$best_idx],
              res$lam_grid[res$idx_1se], res$edf_1se, res$mean_nll[res$idx_1se]))

  # Plot 10-fold CV NLL (+/- 1 SE) curve with vertical markers at lambda_min and lambda_1SE
  log_lam <- log10(res$lam_grid)
  yrng    <- range(c(res$mean_nll - res$se_nll, res$mean_nll + res$se_nll), na.rm = TRUE)
  plot(log_lam, res$mean_nll, type = "b", pch = 19, col = "navy", ylim = yrng,
       xlab = expression(log[10](lambda)), ylab = "10-Fold CV NLL", main = cfg$label)
  segments(log_lam, res$mean_nll - res$se_nll, log_lam, res$mean_nll + res$se_nll, col = "cornflowerblue")
  abline(h = res$mean_nll[res$best_idx] + res$se_nll[res$best_idx], col = "gray60", lty = 3)
  abline(v = log10(res$lam_grid[res$best_idx]), col = "red3", lty = 2, lwd = 1.5)
  abline(v = log10(res$lam_grid[res$idx_1se]),  col = "forestgreen", lty = 4, lwd = 1.5)
  legend("topleft", legend = c("lambda_min", "lambda_1SE"),
         col = c("red3", "forestgreen"), lty = c(2, 4), lwd = 1.5, bty = "n", cex = 0.85)
}
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))



# =====================================================================
# 4. OUT-OF-SAMPLE EVALUATION
# =====================================================================
cat("\n=== SECTION 4: OUT-OF-SAMPLE EVALUATION ===\n")

# ---------------------------------------------------------------------
# 4A. NON-REGULARIZED MODELS: STANDARD 10-FOLD STRATIFIED CV
# ---------------------------------------------------------------------
# Multinomial, POM Logit/Probit, and Ordinal GAM have no CV-tuned hyperparameters
# (GAM estimates its spline smoothing penalties internally on df_tr via REML).
# For each fold k = 1..10: fit strictly on the 9/10 training slice df_tr and
# predict probabilities once on the untouched 1/10 test slice df_te.
cat("\n=== SECTION 4A: NON-REGULARIZED MODELS (10-FOLD CV) ===\n")
oof_multinom   <- matrix(NA_real_, n_obs, n_classes)
oof_pom_logit  <- matrix(NA_real_, n_obs, n_classes)
oof_pom_probit <- matrix(NA_real_, n_obs, n_classes)
oof_gam        <- matrix(NA_real_, n_obs, n_classes)

for (k in seq_len(K_folds)) {
  te    <- fold_list[[k]]
  tr    <- setdiff(seq_len(n_obs), te)
  df_tr <- df_model[tr, ]
  df_te <- df_model[te, ]

  # 0. Multinomial Logit (requires an unordered factor response)
  df_tr_nom <- df_tr; df_tr_nom$Overall.Qual <- factor(df_tr$Overall.Qual, ordered = FALSE)
  df_te_nom <- df_te; df_te_nom$Overall.Qual <- factor(df_te$Overall.Qual, ordered = FALSE)
  fit_m <- multinom(Overall.Qual ~ ., data = df_tr_nom, maxit = 1000, MaxNWts = 5000, trace = FALSE)
  oof_multinom[te, ] <- predict(fit_m, newdata = df_te_nom, type = "probs")

  # 1a / 1b. Cumulative Proportional Odds Logit and Probit (Hess = FALSE skips SE overhead)
  fit_pl <- polr(Overall.Qual ~ ., data = df_tr, method = "logistic", Hess = FALSE)
  fit_pp <- polr(Overall.Qual ~ ., data = df_tr, method = "probit",   Hess = FALSE)
  oof_pom_logit[te, ]  <- predict(fit_pl, newdata = df_te, type = "probs")
  oof_pom_probit[te, ] <- predict(fit_pp, newdata = df_te, type = "probs")

  # 2. Ordinal GAM (ocat expects integer response 1..K; REML runs strictly inside df_tr)
  df_tr_gam <- df_tr; df_tr_gam$Overall_Qual_int <- as.integer(df_tr$Overall.Qual)
  df_te_gam <- df_te; df_te_gam$Overall_Qual_int <- as.integer(df_te$Overall.Qual)
  fit_g <- gam(gam_formula, family = ocat(R = n_classes), data = df_tr_gam, method = "REML")
  oof_gam[te, ] <- predict(fit_g, newdata = df_te_gam, type = "response")
}

# ---------------------------------------------------------------------
# 4B. REGULARIZED MODELS: NESTED 10-FOLD STRATIFIED CV (10 OUTER x 5 INNER)
# ---------------------------------------------------------------------
# Because lambda is selected via CV, evaluating lambda chosen on all 10 folds would
# leak validation information and underestimate test error. Instead, for every outer
# fold k = 1..10:
#   1) Hold out te (test) completely untouched.
#   2) Split ONLY the 9/10 training matrix X_tr into K_inner = 5 stratified inner folds.
#   3) Select fold-specific lambda_min and lambda_1SE inside tr (training) via tune_onet().
#   4) Predict te once using the model refitted on all of tr at those inner-tuned lambdas.
cat("\n=== SECTION 4B: REGULARIZED MODELS (NESTED 10x5 CV) ===\n")
oof_lasso_min <- matrix(NA_real_, n_obs, n_classes)
oof_lasso_1se <- matrix(NA_real_, n_obs, n_classes)
oof_enet_min  <- matrix(NA_real_, n_obs, n_classes)
oof_enet_1se  <- matrix(NA_real_, n_obs, n_classes)
oof_ridge_min <- matrix(NA_real_, n_obs, n_classes)
oof_ridge_1se <- matrix(NA_real_, n_obs, n_classes)

for (k in seq_len(K_folds)) {
  te          <- fold_list[[k]]
  tr          <- setdiff(seq_len(n_obs), te)
  X_tr        <- X_all[tr, , drop = FALSE]
  X_te        <- X_all[te, , drop = FALSE]
  y_tr        <- y_all[tr]
  inner_folds <- make_folds(y_tr, K_inner, seed = 100 + k)$fold_list

  # Tune Lasso (alpha = 1), ElasticNet (alpha = 0.5), and Ridge (alpha = 0) strictly inside tr
  tn_las_k <- tune_onet(X_tr, y_tr, inner_folds, alpha = 1.0)
  tn_ent_k <- tune_onet(X_tr, y_tr, inner_folds, alpha = 0.5)
  tn_rdg_k <- tune_onet(X_tr, y_tr, inner_folds, alpha = 0.0)

  # Record out-of-fold predictions on te for both usual (min) and parsimonious (+1 SE) rules
  oof_lasso_min[te, ] <- predict(tn_las_k$fit_full, newx = X_te, whichLambda = tn_las_k$best_idx, type = "response")
  oof_lasso_1se[te, ] <- predict(tn_las_k$fit_full, newx = X_te, whichLambda = tn_las_k$idx_1se,  type = "response")
  oof_enet_min[te, ]  <- predict(tn_ent_k$fit_full, newx = X_te, whichLambda = tn_ent_k$best_idx, type = "response")
  oof_enet_1se[te, ]  <- predict(tn_ent_k$fit_full, newx = X_te, whichLambda = tn_ent_k$idx_1se,  type = "response")
  oof_ridge_min[te, ] <- predict(tn_rdg_k$fit_full, newx = X_te, whichLambda = tn_rdg_k$best_idx, type = "response")
  oof_ridge_1se[te, ] <- predict(tn_rdg_k$fit_full, newx = X_te, whichLambda = tn_rdg_k$idx_1se,  type = "response")
}


# =====================================================================
# 5. SUMMARY TABLES & DIAGNOSTIC PLOTS
# =====================================================================

# Helper 1: replace tiny/zero predicted probabilities with 1e-15 so log() never returns -Inf,
# then renormalize each observation's predicted class probabilities to sum to 1
clip_probs <- function(P) {
  P <- pmax(P, 1e-15)
  sweep(P, 1, rowSums(P), "/")
}

# Helper 2: compute all full-sample diagnostics (Train_NLL, AIC, BIC) and out-of-fold
# cross-validation metrics (CV_NLL, SE_NLL, Overfit_Gap, Accuracy, MAE_Class) for one model
score_model <- function(name, y, prob_train, prob_oof, edf) {
  y_idx  <- as.integer(y)
  P_tr   <- clip_probs(prob_train)
  P_cv   <- clip_probs(prob_oof)
  nll_tr <- -log(P_tr[cbind(seq_len(n_obs), y_idx)])  # in-sample log-loss per home
  nll_cv <- -log(P_cv[cbind(seq_len(n_obs), y_idx)])  # out-of-fold log-loss per home
  ll_tr  <- -sum(nll_tr)                              # full-sample maximized log-likelihood
  y_hat  <- max.col(P_cv, ties.method = "first")      # MAP predicted quality tier (1..K)
  f_nll  <- tapply(nll_cv, fold_id, mean)             # fold-level mean CV NLL for SE across 10 folds

  data.frame(
    Model        = name,
    EDF          = round(edf, 1),
    Train_LogLik = round(ll_tr, 2),
    Train_NLL    = round(mean(nll_tr), 4),
    CV_LogLik    = round(-sum(nll_cv), 2),
    CV_NLL       = round(mean(nll_cv), 4),
    SE_NLL       = round(sd(f_nll) / sqrt(K_folds), 4),
    Overfit_Gap  = round(mean(nll_cv) - mean(nll_tr), 4),
    AIC          = round(-2 * ll_tr + 2 * edf, 2),
    BIC          = round(-2 * ll_tr + log(n_obs) * edf, 2),
    Accuracy     = round(mean(y_hat == y_idx), 4),
    MAE_Class    = round(mean(abs(y_hat - y_idx)), 4),
    Neg_Probs    = sum(prob_oof < -1e-6, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

# Assemble and rank all 10 benchmarked models (including both min and +1 SE rules for penalized fits)
edf_pom <- length(c(coef(model_pom_logit), model_pom_logit$zeta))
results_full <- rbind(
  score_model("0. Multinomial (Unordered)",       y_all, prob_multinom,                    oof_multinom,  model_multinom$edf),
  score_model("1a. POM Logit (polr)",             y_all, prob_pom_logit,                   oof_pom_logit, edf_pom),
  score_model("1b. POM Probit (polr)",            y_all, prob_pom_probit,                  oof_pom_probit,edf_pom),
  score_model("2. Ordinal GAM (mgcv ocat)",       y_all, prob_gam,                         oof_gam,       attr(logLik(model_gam), "df")),
  score_model("4a. ordinalNet Lasso (min)",       y_all, onet_results$lasso$prob_full_min, oof_lasso_min, onet_results$lasso$edf_min),
  score_model("4b. ordinalNet Lasso (1SE)",       y_all, onet_results$lasso$prob_full_1se, oof_lasso_1se, onet_results$lasso$edf_1se),
  score_model("4c. ordinalNet ElasticNet (min)",  y_all, onet_results$enet$prob_full_min,  oof_enet_min,  onet_results$enet$edf_min),
  score_model("4d. ordinalNet ElasticNet (1SE)",  y_all, onet_results$enet$prob_full_1se,  oof_enet_1se,  onet_results$enet$edf_1se),
  score_model("4e. ordinalNet Ridge (min)",       y_all, onet_results$ridge$prob_full_min, oof_ridge_min, onet_results$ridge$edf_min),
  score_model("4f. ordinalNet Ridge (1SE)",       y_all, onet_results$ridge$prob_full_1se, oof_ridge_1se, onet_results$ridge$edf_1se)
)
results_full <- results_full[order(results_full$CV_NLL), ]

# --- TABLE 1: Master Model Comparison (Metrics 1, 2, 3) ---
cat("\n=== TABLE 1: MASTER MODEL COMPARISON (SORTED BY CV_NLL) ===\n")
print(results_full[, c("Model", "EDF", "Train_NLL", "CV_NLL", "SE_NLL",
                       "Overfit_Gap", "AIC", "BIC", "Accuracy", "MAE_Class")], row.names = FALSE)
cat("Sanity check - negative OOF probability entries:", sum(results_full$Neg_Probs), "\n")

# --- TABLE 2: Fitted Slopes (min vs +1 SE) & GAM Smooth Complexity ---
# GAM_EDF = 1.00 indicates a purely linear term; GAM_EDF > 1 measures spline nonlinearity
gam_smry  <- summary(model_gam)
gam_edf_v <- sapply(feat_names, function(v) {
  sn <- paste0("s(", v, ")")
  if (sn %in% rownames(gam_smry$s.table)) gam_smry$s.table[sn, "edf"] else 1.0
})
cat("\n=== TABLE 2: SLOPES, SHRINKAGE (MIN vs 1-SE) & GAM EDF ===\n")
print(data.frame(
  Predictor  = feat_names,
  POM_Logit  = round(coef(model_pom_logit)[feat_names], 3),
  POM_Probit = round(coef(model_pom_probit)[feat_names], 3),
  Lasso_min  = round(onet_results$lasso$b_min, 3), Lasso_1SE = round(onet_results$lasso$b_1se, 3),
  ENet_min   = round(onet_results$enet$b_min,  3), ENet_1SE  = round(onet_results$enet$b_1se,  3),
  Ridge_min  = round(onet_results$ridge$b_min, 3), Ridge_1SE = round(onet_results$ridge$b_1se, 3),
  GAM_EDF    = round(gam_edf_v, 2),
  stringsAsFactors = FALSE
), row.names = FALSE)

# --- TABLE 3: Per-Class Out-of-Fold Recall Across Merged Quality Tiers ---
# Checks how often each model correctly identifies individual tiers (especially rare tails "<=3" and ">=9")
oof_list <- list(
  Multinom  = oof_multinom,  POM_Logit = oof_pom_logit, Ord_GAM   = oof_gam,
  Lasso_min = oof_lasso_min, Lasso_1SE = oof_lasso_1se,
  ENet_min  = oof_enet_min,  ENet_1SE  = oof_enet_1se,
  Ridge_min = oof_ridge_min, Ridge_1SE = oof_ridge_1se
)
y_true_idx <- as.integer(y_all)
table_recall <- data.frame(Overall_Qual = qual_levels, Count = as.integer(table(y_all)), stringsAsFactors = FALSE)
for (m_lbl in names(oof_list)) {
  y_pred <- max.col(clip_probs(oof_list[[m_lbl]]), ties.method = "first")
  table_recall[[m_lbl]] <- round(sapply(seq_len(n_classes), function(c) mean(y_pred[y_true_idx == c] == c)), 3)
}
cat("\n=== TABLE 3: PER-CLASS OUT-OF-FOLD RECALL ===\n")
print(table_recall, row.names = FALSE)


# --- FIGURES 1 & 2: Clean Bar Charts of All 3 Metrics + Overfit Gap ---
# Green = Ordinal GAM, blue = POM / penalized POM, red = Unordered Multinomial
fam_cols <- ifelse(grepl("Ordinal GAM", results_full$Model), "forestgreen",
            ifelse(grepl("Multinomial", results_full$Model), "firebrick", "royalblue3"))

# Concise labels matching results_full row order (best CV_NLL at top of each chart)
short_names <- gsub("ordinalNet ", "", results_full$Model)
ord_idx     <- rev(seq_len(nrow(results_full)))

# Reusable horizontal barplot helper printing each bar's exact value at the right tip
plot_hbar <- function(vals, labels, cols, title, xlab, fmt = "%.3f") {
  v_ord <- vals[ord_idx]
  bp <- barplot(v_ord, names.arg = labels[ord_idx], horiz = TRUE, las = 1,
                col = cols[ord_idx], border = NA, xlim = c(0, max(v_ord) * 1.18),
                main = title, xlab = xlab, cex.main = 0.96, cex.names = 0.82, cex.axis = 0.82)
  text(v_ord, bp, labels = sprintf(fmt, v_ord), pos = 4, cex = 0.78)
}

# Figure 1: Likelihood & Overfitting (2 stacked full-width charts for wide, readable bars)
par(mfrow = c(2, 1), mar = c(4.0, 13.0, 2.8, 1.5))
plot_hbar(results_full$CV_NLL,      short_names, fam_cols, "Fig 1A: CV NLL (Lower = Better)",      "Out-of-Fold NLL",   "%.3f")
plot_hbar(results_full$Overfit_Gap, short_names, fam_cols, "Fig 1B: Overfit Gap (Lower = Better)", "CV_NLL - Train_NLL", "%.3f")

# Figure 2: Discrete Classification Accuracy & Tier MAE (same row order as Figure 1)
par(mfrow = c(2, 1), mar = c(4.0, 13.0, 2.8, 1.5))
plot_hbar(100 * results_full$Accuracy, short_names, fam_cols, "Fig 2A: Accuracy (Higher = Better)", "Accuracy (%)",    "%.1f%%")
plot_hbar(results_full$MAE_Class,      short_names, fam_cols, "Fig 2B: Tier MAE (Lower = Better)",  "MAE_Class (Tiers)", "%.3f")


# --- FIGURE 3: Standardized Feature Effects (POM Logit vs Lasso min & 1-SE) ---
# Multiplying each slope by its predictor's SD puts square footage, years, and counts on one comparable scale
par(mfrow = c(1, 1), mar = c(4.6, 10.0, 3.2, 2.0))
sd_x <- apply(X_all, 2, sd)
std_mat <- rbind(
  POM_Logit = coef(model_pom_logit)[feat_names] * sd_x,
  Lasso_min = onet_results$lasso$b_min * sd_x,
  Lasso_1SE = onet_results$lasso$b_1se * sd_x
)
barplot(std_mat[, rev(seq_along(feat_names))], beside = TRUE, horiz = TRUE, las = 1,
        xlim = range(c(-0.55, 1.25, std_mat * 1.12)),
        col = c("royalblue3", "darkorange2", "forestgreen"), border = NA,
        xlab = "Effect per 1-SD Increase in Predictor (Log-Odds Scale)",
        main = "Fig 3: Standardized Effects & Lasso 1-SE Variable Selection",
        cex.main = 0.96, cex.names = 0.78)
abline(v = 0, col = "gray40", lty = 2)
legend("right", legend = c("POM Logit", "Lasso (min)", "Lasso (1SE)"),
       fill = c("royalblue3", "darkorange2", "forestgreen"), border = NA, bty = "n", cex = 0.85)


# --- FIGURE 4: Out-of-Fold Confusion Matrices Across Key Model Families ---
# Square 2x2 grid comparing unordered baseline, POM, sparse Lasso (1SE), and Ordinal GAM
plot_confusion <- function(P_oof, title) {
  y_hat <- factor(qual_levels[max.col(clip_probs(P_oof), ties.method = "first")], levels = qual_levels)
  cm    <- as.matrix(table(True = y_all, Pred = y_hat))
  cm_n  <- sweep(cm, 1, pmax(rowSums(cm), 1), "/")  # normalize each true tier row to sum to 1
  pal   <- colorRampPalette(c("aliceblue", "steelblue3", "navy"))(100)

  image(seq_len(n_classes), seq_len(n_classes), t(cm_n[n_classes:1, ]), col = pal, zlim = c(0, 1),
        axes = FALSE, xlab = "Predicted Tier", ylab = "True Tier",
        main = title, cex.main = 0.92)
  axis(1, at = seq_len(n_classes), labels = qual_levels, cex.axis = 0.76)
  axis(2, at = seq_len(n_classes), labels = rev(qual_levels), las = 1, cex.axis = 0.76)
  for (r in seq_len(n_classes)) {
    for (c in seq_len(n_classes)) {
      val <- cm_n[r, c]
      if (val >= 0.08) {
        text(c, n_classes - r + 1, sprintf("%.0f%%", 100 * val),
             col = ifelse(val > 0.45, "white", "black"), cex = 0.68)
      }
    }
  }
}

par(mfrow = c(2, 2), pty = "s", mar = c(3.8, 3.8, 2.6, 0.8))
plot_confusion(oof_multinom,  "Fig 4A: 0. Multinomial")
plot_confusion(oof_pom_logit, "Fig 4B: 1a. POM Logit")
plot_confusion(oof_lasso_1se, "Fig 4C: 4b. Lasso (1SE)")
plot_confusion(oof_gam,       "Fig 4D: 2. Ordinal GAM")
par(mfrow = c(1, 1), pty = "m", mar = c(5.1, 4.1, 4.1, 2.1))
