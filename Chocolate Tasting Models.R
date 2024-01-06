# Read data ----

library(tidyverse)

url <- "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-18/chocolate.csv"

chocolate <- read_csv(url)

# EDA ----

chocolate |>
  ggplot(mapping = aes(x = rating)) +
  geom_histogram(bins = 15) +
  theme_minimal()

# Mostly 3 stars

library(tidytext)

# Item tokenization
tidy_chocolate <- chocolate |>
  unnest_tokens(word, most_memorable_characteristics)

# Most common words
tidy_chocolate |>
  count(word, sort = TRUE)

# Most common words by rating
tidy_chocolate |>
  group_by(word) |>
  summarize(n = n(),
            rating = mean(rating)) |>
  ggplot(mapping = aes(x = n, y = rating)) +
  geom_hline(yintercept = mean(chocolate$rating),
             lty = 2,
             color = "gray50") +
  geom_point() +
  geom_text(mapping = aes(label = word),
            check_overlap = TRUE,
            vjust = "top",
            hjust = "left") +
  scale_x_log10()

# Split the data ----

library(tidymodels)

set.seed(123)
chocolate_split <- initial_split(chocolate, strata = rating)
chocolate_train <- training(chocolate_split)
chocolate_test <- testing(chocolate_split)

# Cross validation
set.seed(123)
chocolate_folds <- vfold_cv(chocolate_train, strata = rating)

# Preprocess ----

library(textrecipes)

# Build recipe
chocolate_recipe <- recipe(rating ~ most_memorable_characteristics, data = chocolate_train) |>
  step_tokenize(most_memorable_characteristics) |>
  step_tokenfilter(most_memorable_characteristics, max_tokens = 100) |>
  step_tf(most_memorable_characteristics)

# Let's look at the data this will output
# Prep is like fit, bake is like predict
prep(chocolate_recipe) |>
  bake(new_data = NULL)

# Build models ----

# Each model in tidymodels has algorithm, engine (can use spark), and mode (classification, regression)

ranger_spec <- rand_forest(trees = 500) |>
  set_engine("ranger") |>
  set_mode("regression")

svm_spec <- svm_linear() |>
  set_engine("LiblineaR") |>
  set_mode("regression")

# Build workflow (preprocessing + spec) ----

ranger_wf <- workflow(chocolate_recipe, ranger_spec)
svm_wf <- workflow(chocolate_recipe, svm_spec)

# Evaluate models ----

doParallel::registerDoParallel()

control_preds <- control_resamples(save_pred = TRUE)

svm_rs <- fit_resamples(
  svm_wf,
  resamples = chocolate_folds,
  control = control_preds
)

ranger_rs <- fit_resamples(
  ranger_wf,
  resamples = chocolate_folds,
  control = control_preds
)

# R^2 and RMSE
collect_metrics(svm_rs)
collect_metrics(ranger_rs)

# Plot predictions
bind_rows(
  collect_predictions(svm_rs) |>
    mutate(mod = "SVM"),
  collect_predictions(ranger_rs) |>
    mutate(mod = "ranger")
) |>
  ggplot(mapping = aes(x = rating, y = .pred, color = id)) +
  geom_abline(lty = 2, color = "gray50") +
  geom_jitter(alpha = 0.5) +
  facet_wrap(~mod) +
  coord_fixed()

# Final model ----
final_fitted <- last_fit(svm_wf, chocolate_split)
collect_metrics(final_fitted)

final_wf <- extract_workflow(final_fitted)
predict(final_wf, chocolate_test[55,])

final_wf |>
  tidy() |>
  filter(term != "Bias") |>
  group_by(estimate > 0) |>
  slice_max(abs(estimate), n = 10) |>
  ungroup() |>
  mutate(term = str_remove(term, "tf_most_memorable_characteristics_")) |>
  ggplot(mapping = aes(x = estimate,
                       y = fct_reorder(term, estimate),
                       fill = estimate > 0)) +
  geom_col() +
  theme_minimal()
