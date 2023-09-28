if (!require(glmnet)) {
    quit("no", 2)
}

library(glmnet)

# command args
temp_dir <- NULL
gamma <- 1
seed <- NULL

args <- commandArgs(trailingOnly=TRUE)
for (arg in args) {
    s <- strsplit(arg, "=")[[1]]
    if (length(s) < 2) {
        next
    }

    if (s[1] == "--temp_dir") {
        temp_dir <- paste(s[2:length(s)], collapse="=")
    } else if (s[1] == "--gamma") {
        gamma <- as.numeric(s[2])
    } else if (s[1] == "--rs") {
        seed <- strtoi(ints[2])
    }
}

set.seed(seed)

# function args
path <- file.path(temp_dir, "X.csv")
X <- read.csv(path, sep=',', header=FALSE)

path <- file.path(temp_dir, "y.csv")
y <- read.csv(path, sep=',', header=FALSE)

# calculate penalty_score
fit <- glmnet(
    as.matrix(X),
    as.matrix(y),
    lambda = 0,
    family = "binomial",
    standardize=FALSE,
)
coef <- coef(fit, s=fit$lmabda.min)
coef <- as.matrix(coef)[-1, ]
penalty_factor <- 1 / (abs(coef) ** gamma)

# fit
fit <- glmnet(
    as.matrix(X),
    as.matrix(y),
    alpha = 1,
    penalty.factor = penalty_factor,
    family = "binomial",
    standardize=FALSE,
)

# bic
deviance <- deviance(fit)
k <- fit$df
n <- fit$nobs
bic <- log(n) * k + deviance

# select
best_index <- which.min(bic)

# result
best_lambda <- fit$lambda[best_index]
best_coefs <- coef(fit, s=best_lambda)

# output
path <- file.path(temp_dir, "coefs.csv")
write.csv(as.matrix(best_coefs), path)

quit("no", 0)
