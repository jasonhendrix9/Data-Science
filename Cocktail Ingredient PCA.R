library(tidyverse)
library(janitor)
library(tidymodels)
library(tidytext)

df <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-26/boston_cocktails.csv")

df |>
  count(ingredient, sort = TRUE)

# Notice that the data is very messy

# Data cleaning ----
df <- df |>
  mutate(ingredient = str_to_lower(ingredient),
         ingredient = str_replace_all(ingredient, "-", " "),
         ingredient = str_remove(ingredient, " liqueur"),
         ingredient = str_remove(ingredient, " (if desired)"),
         ingredient = case_when(str_detect(ingredient, "bitters") ~ "bitters",
                                str_detect(ingredient, "orange") ~ "orange juice",
                                str_detect(ingredient, "lemon") ~ "lemon juice",
                                str_detect(ingredient, "lime") ~ "lime juice",
                                str_detect(ingredient, "grapefruit") ~ "grapefruit juice",
                                TRUE ~ ingredient),
         # For bitters, the unit is probably a dash and not an ounce
         measure = case_when(str_detect(ingredient, "bitters") ~ str_replace(measure, "oz$", "dash"),
                             TRUE ~ measure),
         # Convert factions to decimals
         measure = str_replace(measure, " ?1/2", ".5"),
         measure = str_replace(measure, " ?1/4", ".25"),
         measure = str_replace(measure, " ?3/4", ".75"),
         measure_number = parse_number(measure),
         # If the measure is a dash, call it 1/50 of an ounce
         measure_number = if_else(str_detect(measure, "dash"), measure_number / 50, measure_number)) |>
  add_count(ingredient) |>
  filter(n >= 15) |>
  select(-n) |>
  distinct(row_id, ingredient, .keep_all = TRUE) |>
  na.omit()

# Pivot measure numbers based on ingredient and clean names
df <- df |>
  select(-ingredient_number, -row_id, -measure) |>
  pivot_wider(names_from = ingredient, values_from = measure_number, values_fill = 0) |>
  clean_names()

# The data is now much cleaner, albeit not perfect

# PCA ----
pca_recipe <- recipe(~ ., data = df) |>
  # name + category = id
  update_role(name, category, new_role = "id") |>
  step_normalize(all_predictors()) |>
  step_pca(all_predictors())

# Perform the PCA
pca_prep <- prep(pca_recipe)

# Tidy the results
pca_tidied <- tidy(pca_prep, 2)

# Look at the contributions of each ingredient to the first 5 components
pca_tidied |>
  filter(component %in% paste0("PC", 1:5)) |>
  mutate(component = fct_inorder(component)) |>
  ggplot(mapping = aes(x = value, y = terms, fill = terms)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~component, nrow = 1) +
  labs(y = NULL)

# Look at the top 8 contributors to each of the first 4 components
pca_tidied |>
  filter(component %in% paste0("PC", 1:4)) |>
  group_by(component) |>
  top_n(8, abs(value)) |>
  ungroup() |>
  mutate(terms = reorder_within(terms, abs(value), component)) |>
  ggplot(mapping = aes(x = abs(value), y = terms, fill = value > 0)) +
  geom_col() +
  scale_y_reordered() +
  facet_wrap(~component, scales = "free_y") +
  labs(y = NULL, fill = "Positive?")

# Where are the name and categories are in the 5-D space?
juice(pca_prep)

juice(pca_prep) |>
  ggplot(mapping = aes(PC1, PC2, label = name)) +
  geom_point(mapping = aes(color = category)) +
  geom_text(check_overlap = TRUE, hjust = "inward") +
  labs(color = NULL)
