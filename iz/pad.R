library(rmetalog)

my_metalog <- metalog(fishSize$FishSize,
    term_limit = 4,
    term_lower_bound = 4,
    bounds = c(0),
    boundedness = "sl",
    step_len = .01,
    fit_method = "OLS",
    save_data = TRUE
)

my_metalog$A
