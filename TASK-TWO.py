import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data.csv"          
OUTPUT_DIR = "eda_output"       
TOP_K_CATEGORIES = 10         
HIST_BINS = 25         


# ============================================================
# UTILS
# ============================================================
def ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def save_fig(path: str):
    """Save current matplotlib fig nicely."""
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns (safe)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols]


# ============================================================
# DATA LOADING
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    print_section("STEP 1: LOAD DATASET")
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset from: {path}")
    return df


# ============================================================
# BASIC OVERVIEW
# ============================================================
def basic_overview(df: pd.DataFrame):
    print_section("STEP 2: BASIC OVERVIEW")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn Names:")
    print(list(df.columns))

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 Rows:")
    print(df.head())


# ============================================================
# QUALITY CHECKS
# ============================================================
def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    print_section("STEP 3: MISSING VALUES REPORT")

    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    report = (
        pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
        .sort_values("missing_count", ascending=False)
    )

    non_zero = report[report["missing_count"] > 0]

    if len(non_zero) == 0:
        print("✅ No missing values found.")
    else:
        print("⚠ Missing values detected (Top rows):")
        print(non_zero.head(20))

    return report


def duplicate_rows_report(df: pd.DataFrame) -> int:
    print_section("STEP 4: DUPLICATE ROWS REPORT")
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")

    if dup_count > 0:
        print("Suggestion: Remove duplicates using df.drop_duplicates()")

    return dup_count


# ============================================================
# COLUMN TYPE SEPARATION
# ============================================================
def split_columns(df: pd.DataFrame):
    print_section("STEP 5: COLUMN TYPE SPLIT")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Datetime columns ({len(datetime_cols)}): {datetime_cols}")

    return numeric_cols, categorical_cols, datetime_cols


# ============================================================
# SUMMARY STATISTICS
# ============================================================
def summary_statistics(df: pd.DataFrame, numeric_cols, categorical_cols):
    print_section("STEP 6: SUMMARY STATISTICS")

    if len(numeric_cols) > 0:
        print("\n✅ Numerical Summary:")
        print(df[numeric_cols].describe().T)

    if len(categorical_cols) > 0:
        print("\n✅ Categorical Summary:")
        cat_summary = df[categorical_cols].describe().T
        print(cat_summary)


# ============================================================
# OUTLIER REPORT (IQR METHOD)
# ============================================================
def outlier_report_iqr(df: pd.DataFrame, numeric_cols) -> pd.DataFrame:
    print_section("STEP 7: OUTLIER REPORT (IQR METHOD)")

    if len(numeric_cols) == 0:
        print("No numeric columns found, skipping outlier report.")
        return pd.DataFrame()

    rows = []
    for col in numeric_cols:
        series = df[col].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = series[(series < lower) | (series > upper)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(series)) * 100

        rows.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct
        })

    report = pd.DataFrame(rows).sort_values("outlier_count", ascending=False)

    if report.empty:
        print("No outlier report generated.")
    else:
        print("Top potential outliers columns:")
        print(report.head(15))

    return report


# ============================================================
# PLOTS
# ============================================================
def plot_histograms(df: pd.DataFrame, numeric_cols: list, output_dir: str):
    print_section("STEP 8: HISTOGRAMS (NUMERIC FEATURES)")

    if len(numeric_cols) == 0:
        print("No numeric columns found, skipping histograms.")
        return

    df[numeric_cols].hist(figsize=(14, 10), bins=HIST_BINS)
    plt.suptitle("Histograms of Numerical Features")
    save_fig(os.path.join(output_dir, "histograms_numeric.png"))
    print("✅ Saved: histograms_numeric.png")

def plot_boxplots(df: pd.DataFrame, numeric_cols: list, output_dir: str):
    print_section("STEP 9: BOXPLOTS (OUTLIER VISUAL CHECK)")

    if len(numeric_cols) == 0:
        print("No numeric columns found, skipping boxplots.")
        return

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        plt.figure(figsize=(8, 3))
        plt.boxplot(series, vert=False)
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)

        file_name = f"boxplot_{col}.png".replace("/", "_")
        save_fig(os.path.join(output_dir, file_name))

    print("✅ Saved boxplots for numeric columns.")


def plot_categorical_barcharts(df: pd.DataFrame, categorical_cols: list, output_dir: str, top_k: int = 10):
    print_section("STEP 10: BAR CHARTS (CATEGORICAL FEATURES)")

    if len(categorical_cols) == 0:
        print("No categorical columns found, skipping bar charts.")
        return

    for col in categorical_cols:
        vc = df[col].astype(str).value_counts().head(top_k)

        plt.figure(figsize=(10, 4))
        vc.plot(kind="bar")
        plt.title(f"Top {top_k} Categories in {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        file_name = f"bar_{col}.png".replace("/", "_")
        save_fig(os.path.join(output_dir, file_name))

    print("✅ Saved bar charts for categorical columns.")


def correlation_analysis(df: pd.DataFrame, numeric_cols: list, output_dir: str):
    print_section("STEP 11: CORRELATION ANALYSIS")

    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation.")
        return None, None

    corr = df[numeric_cols].corr()

    corr.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    print("✅ Saved correlation_matrix.csv")

    plt.figure(figsize=(10, 7))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    save_fig(os.path.join(output_dir, "correlation_heatmap.png"))
    print("✅ Saved correlation_heatmap.png")

    corr_pairs = (
        corr.where(~np.eye(corr.shape[0], dtype=bool))  
        .stack()
        .sort_values(ascending=False)
    )

    top_pos = corr_pairs.head(10)
    top_neg = corr_pairs.tail(10)

    top_pos.to_csv(os.path.join(output_dir, "top_positive_correlations.csv"))
    top_neg.to_csv(os.path.join(output_dir, "top_negative_correlations.csv"))

    print("\nTop Positive Correlations:")
    print(top_pos)

    print("\nTop Negative Correlations:")
    print(top_neg)

    return top_pos, top_neg


# ============================================================
# EXPORT TABLE REPORTS
# ============================================================
def export_reports(output_dir: str, missing_report, outlier_report):
    print_section("STEP 12: EXPORT REPORT TABLES")

    missing_report.to_csv(os.path.join(output_dir, "missing_values_report.csv"))
    print("✅ Saved missing_values_report.csv")

    if outlier_report is not None and not outlier_report.empty:
        outlier_report.to_csv(os.path.join(output_dir, "outlier_report_iqr.csv"), index=False)
        print("✅ Saved outlier_report_iqr.csv")


# ============================================================
# MAIN RUNNER
# ============================================================
def run_eda():
    ensure_output_dir(OUTPUT_DIR)

    df = load_data(DATA_PATH)

    basic_overview(df)

    missing_report = missing_values_report(df)
    dup_count = duplicate_rows_report(df)

    numeric_cols, categorical_cols, datetime_cols = split_columns(df)

    summary_statistics(df, numeric_cols, categorical_cols)

    outliers = outlier_report_iqr(df, numeric_cols)

    plot_histograms(df, numeric_cols, OUTPUT_DIR)
    plot_categorical_barcharts(df, categorical_cols, OUTPUT_DIR, top_k=TOP_K_CATEGORIES)
    plot_boxplots(df, numeric_cols, OUTPUT_DIR)
    correlation_analysis(df, numeric_cols, OUTPUT_DIR)

    export_reports(OUTPUT_DIR, missing_report, outliers)

    print_section("✅ EDA COMPLETED")
    print(f"All outputs saved in folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
