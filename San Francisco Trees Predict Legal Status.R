# Libraries ----

library(tidyverse)
library(tidymodels)
library(themis)
library(vip)

# Preprocessing ----

read_tree_data <- function()
{
  url <- "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-28/sf_trees.csv"

  sf_trees <- read_csv(url, show_col_types = FALSE)

  return(sf_trees)
}

prepare_tree_data <- function(df_trees)
{
  df_trees <- df_trees |>
    mutate(legal_status = case_when(legal_status == "DPW Maintained" ~ legal_status,
                                    TRUE ~ "Other"),
           plot_size = parse_number(plot_size)) |>
    select(-address) |>
    na.omit() |>
    mutate_if(is.character, factor)

  return(df_trees)
}

eda_plots <- function(trees_df)
{
  trees_df |>
    ggplot(mapping = aes(x = longitude, y = latitude, color = legal_status)) +
    geom_point(size = 0.5) +
    labs(color = NULL) +
    theme_minimal()

  trees_df |>
    count(legal_status, caretaker) |>
    add_count(caretaker, wt = n, name = "caretaker_count") |>
    filter(caretaker_count >= 50) |>
    group_by(legal_status) |>
    mutate(percent_legal = n / sum(n)) |>
    ggplot(mapping = aes(x = percent_legal, y = caretaker, fill = legal_status)) +
    geom_col(position = "dodge") +
    theme_minimal()
}

# Build Model ----

build_recipe <- function(trees_train, upsample = FALSE)
{
  if(upsample)
  {
    trees_recipe <- recipe(legal_status ~ ., data = trees_train) |>
      update_role(tree_id, new_role = "id") |>
      step_other(species, caretaker, threshold = 0.01) |>
      step_other(site_info, threshold = 0.005) |>
      step_dummy(all_nominal(), -all_outcomes()) |>
      step_date(date, features = c("year")) |>
      step_rm(date) |>
      step_upsample(legal_status)
  }
  else
  {
    trees_recipe <- recipe(legal_status ~ ., data = trees_train) |>
    update_role(tree_id, new_role = "id") |>
    step_other(species, caretaker, threshold = 0.01) |>
    step_other(site_info, threshold = 0.005) |>
    step_dummy(all_nominal(), -all_outcomes()) |>
    step_date(date, features = c("year")) |>
    step_rm(date) |>
    step_downsample(legal_status)
  }

  return(trees_recipe)
}

build_wf <- function(trees_recipe, model)
{
  tune_wf <- workflow() |>
  add_recipe(trees_recipe) |>
  add_model(model)

  return(tune_wf)
}

# Tune Model ----

tune_with_resamples <- function(tune_wf, trees_folds, rf_grid = 20)
{
  doParallel::registerDoParallel(cores = 8)

  tune_resample <- tune_grid(
    tune_wf,
    resamples = trees_folds,
    grid = rf_grid
  )

  tune_resample |>
  collect_metrics() |>
  filter(.metric == "roc_auc") |>
  select(mean, min_n, mtry) |>
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter") |>
  ggplot(mapping = aes(x = value, y = mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  theme_minimal()

  return(tune_resample)
}

get_best_model <- function(regular_resample, tune_spec, trees_recipe)
{
  best_auc <- select_best(regular_resample, "roc_auc")

  final_rf <- finalize_model(
    tune_spec,
    best_auc
    )

  final_rf |>
  set_engine("ranger", importance = "permutation") |>
  fit(legal_status ~ .,
      data = juice(prep(trees_recipe)) |> select(-tree_id)) |>
  vip(geom = "point") +
  theme_minimal()

  return(final_rf)
}

# Predictions ----

final_model_predictions <- function(final_wf, trees_split, trees_test)
{
  final_result <- final_wf |>
    last_fit(trees_split)

  print(final_result |>
    collect_metrics())

  final_result_predictions <- final_result |>
    collect_predictions()

   final_result_predictions |>
    mutate(correct = case_when(legal_status == .pred_class ~ "Correct",
                             TRUE ~ "Incorrect")) |>
    bind_cols(trees_test) |>
    ggplot(mapping = aes(x = latitude, y = longitude, color = correct)) +
    geom_point() +
    scale_color_manual(values = c("Incorrect" = "red", "Correct" = "forestgreen")) +
    theme_minimal()

  return(final_result_predictions)
}

# Pipeline ----

rf_pipeline <- function(upsample = FALSE)
{
  set.seed(123)

  print("Preprocessing...")

  trees_df <- read_tree_data() |>
    prepare_tree_data()

  eda_plots(trees_df)

  trees_split <- initial_split(trees_df, strata = legal_status)
  trees_train <- training(trees_split)
  trees_test <- testing(trees_split)

  print("Building model...")

  trees_recipe <- build_recipe(trees_train, upsample = upsample)

  print("Tuning model...")

  tune_spec <- rand_forest(
    mtry = tune(),
    trees = 1000,
    min_n = tune()
  ) |>
    set_engine("ranger") |>
    set_mode("classification")

  tune_wf <- build_wf(trees_recipe, tune_spec)
  trees_folds <- vfold_cv(trees_train)
  # tune_resample <- tune_with_resamples(tune_wf, trees_folds, rf_grid = 20)



  rf_grid <- grid_regular(
    mtry(range = c(10, 40)),
    min_n(range = c(2, 8)),
    levels = 5
  )

  regular_resample <- tune_with_resamples(tune_wf, trees_folds, rf_grid = rf_grid)

  final_rf <- get_best_model(regular_resample, tune_spec, trees_recipe)

  print("Making predictions...")

  final_wf <- build_wf(trees_recipe, final_rf)

  predictions <- final_model_predictions(final_wf, trees_split, trees_test)
}

rf_pipeline(upsample = FALSE)
rf_pipeline(upsample = TRUE)

# Upsampling has a 4.7% increase in accuracy and a 0.6% increase in AUC compared
# to downsampling.
