# Load required libraries
# MASS: For proportional odds logit (polr)
# VGAM: For non-proportional odds logit (vglm)
# mgcv: For ordinal generalized additive models (gam)
# kernlab: For kernel transformations and feature expansion

set.seed(42)
N <- 1000


# =====================================================================
# 1. GENERATE DATA FROM AN ORDINAL LOGIT MODEL
# =====================================================================
# Standard ordinal regression models bridge the gap between continuous 
# regression and discrete classification by assuming the existence of an 
# unobserved, continuous latent variable y*.

# Generate features. X2 is designed with a non-linear (parabolic) 
# relationship to test the flexibility of GAM and Kernel methods later.
X1 <- rnorm(N, 0, 1)
X2 <- runif(N, -2, 2)

# Define the true coefficients. In a standard proportional odds model, 
# these slopes are constant across all cutpoints.
true_beta1 <- 1.5
true_beta2 <- -2.0

# Calculate the latent score: y* = X*beta + epsilon
# For an Ordinal Logit model, the error term (epsilon) follows a 
# standard Logistic distribution with mean 0.
epsilon <- rlogis(N, location = 0, scale = 1)
y_star <- true_beta1 * X1 + true_beta2 * (X2^2) + epsilon

# Define strict, ordered thresholds (cutpoints) theta_1 < theta_2.
theta1 <- -3
theta2 <- 1

# Map the continuous latent variable to discrete ordered categories.
Y <- rep(3, N)
Y[y_star <= theta1] <- 1
Y[y_star > theta1 & y_star <= theta2] <- 2
Y <- factor(Y, ordered = TRUE, levels = c(1, 2, 3))

data <- data.frame(Y, X1, X2)




# =====================================================================
# 2. PROPORTIONAL ODDS LOGIT ESTIMATION
# =====================================================================
# The Proportional Odds Model (POM) assumes the slope vector beta is 
# invariant to the choice of category threshold k. It shifts 
# the entire latent distribution uniformly.

library(MASS)
model_pom <- polr(Y ~ X1 + X2, data = data, method = "logistic")
summary(model_pom)


# --- HOW TO INTERPRET THE POM SUMMARY ---
# 1. Coefficients: In MASS::polr, a positive coefficient indicates that as the 
#    predictor increases, the observation is pushed toward *higher* ordered 
#    categories. 
# 2. Proportional Odds Property: Notice there is only one coefficient per 
#    variable (X1, X2). The model assumes this effect is identical whether 
#    transitioning from category 1 to 2, or category 2 to 3.
# 3. Intercepts (1|2, 2|3): These represent the baseline cutpoints (theta_k) 








# =====================================================================
# 3. NON-PROPORTIONAL ODDS LOGIT ESTIMATION
# =====================================================================
# This approach, also known as the generalized ordered logit model, 
# relaxes the proportional odds assumption by allowing different 
# coefficients (w) for each class[cite: 1]. 

library(VGAM)
# Setting parallel = FALSE allows the slopes to vary across thresholds.
model_npom <- vglm(Y ~ X1 + X2, family = cumulative(parallel = FALSE), data = data)
summary(model_npom)


# --- HOW TO INTERPRET THE NPOM SUMMARY ---
# 1. Class-Specific Coefficients: Because we relaxed the proportional odds 
#    assumption, each predictor now has a separate coefficient for each 
#    threshold (e.g., X1:1 and X1:2).
# 2. Diagnosing Proportional Odds: If the proportional odds assumption 
#    were perfectly true, X1:1 would equal X1:2. Significant divergence 
#    between these threshold-specific weights indicates that the standard 
#    POM is too restrictive.










# =====================================================================
# 4. GAM LOGIT (ORDINAL GAM) - CORRECTED
# =====================================================================
# Replaces the rigid linear predictor with an additive sum of smooth, 
# non-linear functions. This is highly effective for capturing 
# non-linear biological parabolas or threshold effects while preserving 
# interpretability.

library(mgcv)

# mgcv's ocat family requires integer class labels instead of factors
data$Y_int <- as.integer(data$Y)

# 'ocat' specifies the ordered categorical family. R sets the number of classes.
model_gam <- gam(Y_int ~ s(X1) + s(X2), family = ocat(R = 3), data = data)
summary(model_gam)

# Visualizing the smooth terms will reveal the quadratic nature of X2
plot(model_gam, pages = 1, shade = TRUE, main = "GAM Smooth Terms")



# --- HOW TO INTERPRET THE GAM SUMMARY ---
# 1. Estimated Degrees of Freedom (edf): This is the key metric for the smooth 
#    terms. An edf close to 1 implies a linear relationship. An edf significantly 
#    greater than 1 indicates a non-linear effect. You should see a 
#    higher edf for X2, correctly identifying the underlying parabola.
# 2. Additive Interpretability: Unlike kernel methods, GAMs preserve full 
#    interpretability because the marginal effects remain additive.
# 3. Cutpoints: The summary will also output the estimated cutpoints dividing 
#    the 3 classes, structurally similar to the POM intercepts.






# =====================================================================
# 5. KERNEL METHOD LOGIT (EXTENDED FEATURE SPACE)
# =====================================================================
# Kernel ordinal regression projects original features into a high-dimensional 
# Reproducing Kernel Hilbert Space (RKHS). This handles complex, 
# non-linear decision boundaries that standard linear links cannot resolve.

library(kernlab)

# We utilize a Radial Basis Function (RBF) / Gaussian kernel, which is a 
# proven positive definite function for RKHS.
rbf <- rbfdot(sigma = 0.5)

# To explicitly extend the feature space and pass it to a logit model, 
# we extract the principal components of the Kernel (Gram) matrix. The Gram 
# matrix encodes the non-linear geometric relationships between points.
kpca_res <- kpca(~ X1 + X2, data = data, kernel = "rbfdot", 
                 kpar = list(sigma = 0.5), features = 15)

# Extract the expanded non-linear feature space
expanded_features <- as.data.frame(predict(kpca_res, data[, c("X1", "X2")]))
colnames(expanded_features) <- paste0("KPC", 1:15)

# Combine the original target with the new expanded feature space
data_kernel <- cbind(data["Y"], expanded_features)

# Fit the proportional odds logit model on the extended kernel features
model_kernel_logit <- polr(Y ~ ., data = data_kernel, method = "logistic")
summary(model_kernel_logit)


# --- HOW TO INTERPRET THE KERNEL LOGIT SUMMARY ---
# 1. Loss of Additive Interpretability: The coefficients (KPC1 to KPC15) 
#    no longer map directly to the original X1 and X2 features. Instead, 
#    they represent the weights of the principal components extracted from 
#    the Reproducing Kernel Hilbert Space (RKHS).
# 2. Modeling Power: By evaluating the model on this Gram matrix projection, 
#    we allow a simple linear logit link to resolve highly complex, non-linear 
#    decision boundaries that standard proportional odds models would fail 
#    to capture.




