import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def read_boston_cocktails_data():
    df = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-26/boston_cocktails.csv")
    return df

def clean_data(df):
    # Take a look at the data to see what needs to be cleaned
    # print(df)

    # Convert ingredient to lowercase and replace hyphens with spaces
    df["ingredient"] = df["ingredient"].str.lower().str.replace("-", " ")

    # Replace " liqueur" and " (if desired)" with
    df["ingredient"] = df["ingredient"].str.replace(" liqueur", "").str.replace(" \\(if desired\\)", "")

    # Simplify the categories (for example, anything containing "bitters" just becomes "bitters")
    df["ingredient"] = np.select(
        [
            df["ingredient"].str.contains("bitters"),
            df["ingredient"].str.contains("orange"),
            df["ingredient"].str.contains("lemon"),
            df["ingredient"].str.contains("lime"),
            df["ingredient"].str.contains("grapefruit")
        ],
        [
            "bitters",
            "orange juice",
            "lemon juice",
            "lime juice",
            "grapefruit juice"
        ],
        default=df["ingredient"]
    )

    return df

def parse_measure_numbers(df):
    # For bitters, replace "oz" with "dash" in measure
    df.loc[df["ingredient"] == "bitters", "measure"] = df["measure"].str.replace("oz$", "dash")

    # Convert fractions to decimals in measure
    df["measure"] = df["measure"].str.replace(" ?1/2", ".5").str.replace(" ?1/4", ".25").str.replace(" ?3/4", ".75")

    # Parse measure_number from measure
    df["measure_number"] = df["measure"].str.extract("(\\d*\\.?\\d+)", expand=False).astype(float)

    # If the measure is a dash, consider it as 1/50 of an ounce
    df.loc[df["measure"].str.contains("dash"), "measure_number"] /= 50

    return df

def shape_top_ingredients(df):
    # Add count column for each ingredient
    df["n"] = df.groupby("ingredient")["ingredient"].transform("count")

    # Filter rows with count greater than or equal to 15
    df = df[df["n"] >= 15]

    # Select relevant columns and keep distinct rows based on row_id and ingredient
    df = (df[["name", "category", "row_id", "ingredient_number", "ingredient", "measure", "measure_number"]].
          drop_duplicates(subset=["row_id", "ingredient"], keep="first"))

    # Drop rows with missing values
    df = df.dropna()

    # Select relevant columns
    df = df.drop(columns=['ingredient_number', 'row_id', 'measure'])

    # Pivot the DataFrame wider
    df = df.pivot_table(index=df.name, columns='ingredient', values='measure_number', fill_value=0)

    # Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    return df

def perform_pca(df):
    pca = PCA(n_components=0.9).fit(df)
    print("95% of the variance:", pca.explained_variance_ratio_)

    df = pd.DataFrame(pca.components_, columns=df.columns).T

    return df

def main():
    df = read_boston_cocktails_data()

    df = clean_data(df)
    df = parse_measure_numbers(df)
    df = shape_top_ingredients(df)

    df = perform_pca(df)

if __name__ == "__main__":
    main()
