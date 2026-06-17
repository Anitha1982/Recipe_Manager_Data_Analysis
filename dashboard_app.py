import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(layout="wide", page_title="Recipe Dataset Analysis")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data" / "raw" / "Recipe_Dataset.csv"

    print(csv_path)
    print(csv_path.exists())

    df = pd.read_csv(csv_path)

    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
    df['CreationYear'] = df['CreatedDate'].dt.year
    df['CreationMonth'] = df['CreatedDate'].dt.month_name()
    return df

recipe_df = load_data()

# Define numerical and categorical columns for reuse
numerical_cols = ['PreparationTime', 'CookingTime', 'TotalTime', 'IngredientCount', 'Calories', 'Rating', 'ReviewCount', 'PopularityScore', 'Cost', 'Protein', 'Fat', 'Carbohydrates', 'Fiber', 'Sugar', 'Sodium', 'Servings']
categorical_cols = ['Cuisine', 'Category', 'Difficulty', 'Region', 'Season', 'Vegetarian', 'Vegan', 'GlutenFree']

# --- Dashboard Title ---
st.title('🍽️ Recipe Dataset: Comprehensive Business Insights 📊')
st.markdown("--- ")

# --- Overarching Insight ---
st.subheader('Overarching Business Insight')
st.markdown(
    """
    The Recipe Dataset reveals a high-quality data source, free from missing values and duplicates, providing a robust foundation for analysis.
    Key opportunities lie in catering to the high demand for easy, time-efficient recipes, especially in popular cuisines and categories.
    Strategic focus on promoting highly-rated recipes and managing numerical outliers can further enhance user engagement and content strategy.
    """
)
st.markdown("--- ")

# --- 2. Integrate Visualizations & Insights ---

st.header('Data Quality & Distribution Insights')

# All Numerical Features Histograms
st.subheader('Distributions of Key Numerical Features')
num_plots_per_row = 3
num_rows = (len(numerical_cols) + num_plots_per_row - 1) // num_plots_per_row

fig = plt.figure(figsize=(num_plots_per_row * 6, num_rows * 4))
for i, col in enumerate(numerical_cols):
    ax = fig.add_subplot(num_rows, num_plots_per_row, i + 1)
    sns.histplot(recipe_df[col], kde=True, bins=30, ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
    **Business Insight: Numerical Feature Distributions**
    *   `PreparationTime`, `CookingTime`, `TotalTime`, `Calories`, `ReviewCount`, `PopularityScore`, and `Cost` show **right-skewed distributions**, indicating a majority of recipes are quick, lower-calorie, less reviewed/popular, and affordable. This suggests opportunities for focusing on these segments or identifying niche markets for the long tail of high-end/complex recipes.
    *   `IngredientCount` appears more **normal-like**, implying a sweet spot for recipe complexity. `Rating` is often **left-skewed**, reflecting user satisfaction with generally good recipes.
    *   The presence of outliers (visible in box plots from previous analysis) in many of these features warrants further investigation for data cleaning or specialized modeling approaches.
    """
)
st.markdown("--- ")

# All Categorical Features Count Plots
st.subheader('Distributions of Categorical Features')
cat_plots_per_row = 2
cat_num_rows = (len(categorical_cols) + cat_plots_per_row - 1) // cat_plots_per_row

fig = plt.figure(figsize=(cat_plots_per_row * 8, cat_num_rows * 5))
for i, col in enumerate(categorical_cols):
    ax = fig.add_subplot(cat_num_rows, cat_plots_per_row, i + 1)
    sns.countplot(y=col, data=recipe_df, order=recipe_df[col].value_counts().index, palette='viridis', ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel('Count')
    ax.set_ylabel(col)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
    **Business Insight: Categorical Feature Distributions**
    *   The dominant `Cuisine` and `Category` types (e.g., Chinese, Dessert) indicate areas of high existing interest and demand. Businesses can leverage this by creating more content in these popular segments or by promoting lesser-represented categories to diversify offerings.
    *   The high frequency of 'Easy' `Difficulty` recipes confirms user preference for simplicity, reinforcing the strategy to focus on accessible content.
    *   Distribution across `Region` and `Season` suggests opportunities for targeted marketing campaigns and seasonal content planning.
    *   High proportions of `Vegetarian`, `Vegan`, and `GlutenFree` options suggest a significant market for dietary-specific recipes, which can be further emphasized in content and search filters.
    """
)
st.markdown("--- ")

# Correlation Heatmap
st.subheader('Correlation Matrix of Numerical Features')
fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(recipe_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix of Numerical Features')
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
    **Business Insight: Numerical Feature Correlations**
    *   Strong positive correlations between `PreparationTime`, `CookingTime`, and `TotalTime` are expected, indicating that these metrics are highly interdependent. This means optimizing one often impacts the others.
    *   Correlations between nutritional components (e.g., `Protein`, `Fat`, `Calories`) provide insights into recipe composition and can guide dietary-focused recipe development.
    *   Weak correlations with `Rating` suggest that user ratings are influenced by a complex interplay of factors beyond simple numerical metrics, potentially involving qualitative aspects or individual preferences. This highlights the challenge in predicting ratings solely from quantitative recipe data.
    *   Positive correlation between `ReviewCount` and `PopularityScore` confirms their aligned role in indicating recipe success, valuable for identifying trending content.
    """
)
st.markdown("--- ")

# Time-Series Trends
st.header('Recipe Creation Trends')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

sns.countplot(x='CreationYear', data=recipe_df, palette='viridis', ax=ax1)
ax1.set_title('Number of Recipes Created Per Year')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Recipes')

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
sns.countplot(y='CreationMonth', data=recipe_df, order=month_order, palette='plasma', ax=ax2)
ax2.set_title('Number of Recipes Created Per Month')
ax2.set_xlabel('Number of Recipes')
ax2.set_ylabel('Month')

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
    **Business Insight: Recipe Creation Trends**
    *   Analyzing recipe creation trends over `CreationYear` and `CreationMonth` can reveal seasonality in content production or user submissions. This information is critical for content planning, resource allocation, and optimizing marketing campaigns.
    *   Spikes or drops in creation over years can indicate platform growth or shifts in content strategy.
    *   Monthly trends can inform seasonal recipe features, ingredient promotions, or themed content initiatives to align with user interest throughout the year.
    """
)
st.markdown("--- ")

# Example 1: PreparationTime Distribution
# This was already added, keeping it for context, but it's now covered by the 'All Numerical Features' section
# st.subheader('Distribution of Preparation Time')
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.histplot(recipe_df['PreparationTime'], kde=True, bins=30, ax=ax)
# ax.set_title('Distribution of Preparation Time')
# ax.set_xlabel('Preparation Time (minutes)')
# ax.set_ylabel('Frequency')
# st.pyplot(fig)
# st.markdown(
#     """
#     **Business Insight:** The right-skewed distribution of `PreparationTime` indicates that most recipes are quick to prepare. Businesses should prioritize and promote these time-efficient recipes to appeal to busy users, while also offering a selection of longer-prep recipes for niche markets or special occasions.
#     """
# )
# st.markdown("--- ")

# Example 2: Rating Distribution by Cuisine
# This was already added, keeping it for context, but it's now covered by the 'Relationships Between Categorical and Numerical Features' section
# st.subheader('Rating Distribution by Cuisine')
# fig, ax = plt.subplots(figsize=(12, 7))
# sns.boxplot(x='Cuisine', y='Rating', data=recipe_df, palette='viridis', ax=ax)
# ax.set_title('Rating Distribution by Cuisine')
# ax.set_xlabel('Cuisine')
# ax.set_ylabel('Rating')
# plt.xticks(rotation=45, ha='right')
# st.pyplot(fig)
# st.markdown(
#     """
#     **Business Insight:** Analyzing rating distributions by cuisine can identify high-performing culinary categories. Focusing on promoting top-rated cuisines or investigating lower-rated ones for improvement can enhance overall user satisfaction and guide content development efforts.
#     """
# )
# st.markdown("--- ")

# Relationships Between Categorical and Numerical Features
st.header('Relationships Between Categorical and Numerical Features')

# Relationship between 'Cuisine' and 'Rating'
st.subheader('Rating Distribution by Cuisine')
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='Cuisine', y='Rating', data=recipe_df, palette='viridis', ax=ax)
ax.set_title('Rating Distribution by Cuisine')
ax.set_xlabel('Cuisine')
ax.set_ylabel('Rating')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
    **Business Insight:** Variations in `Rating` distribution across different `Cuisine` types can highlight which cuisines are consistently well-received or which might need recipe quality improvement. This insight helps in optimizing content strategy to elevate overall user satisfaction and potentially attract new users through popular cuisine offerings.
    """
)
st.markdown("--- ")

# Relationship between 'Difficulty' and 'TotalTime'
st.subheader('Total Time Distribution by Difficulty')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Difficulty', y='TotalTime', data=recipe_df, order=['Easy', 'Medium', 'Hard'], palette='magma', ax=ax)
ax.set_title('Total Time Distribution by Difficulty')
ax.set_xlabel('Difficulty')
ax.set_ylabel('TotalTime (minutes)')
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
    **Business Insight:** The clear trend of increasing `TotalTime` with `Difficulty` confirms the intuitive expectation. This insight is crucial for user experience, allowing for accurate filtering by time constraints and managing expectations. It also reinforces the strategy of categorizing recipes by difficulty to help users find suitable options based on their skills and available time.
    """
)
st.markdown("--- ")

# Relationship between 'Category' and 'Calories'
st.subheader('Calories Distribution by Category')
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='Category', y='Calories', data=recipe_df, palette='cividis', ax=ax)
ax.set_title('Calories Distribution by Category')
ax.set_xlabel('Category')
ax.set_ylabel('Calories')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)
plt.close(fig)
st.markdown(
    """
    **Business Insight:** Differences in `Calories` distribution across `Category` types (e.g., Desserts likely higher, Salads lower) are vital for dietary planning and health-focused recipe recommendations. This can be used to target users with specific dietary needs, highlight healthier options within certain categories, or identify categories where calorie-conscious alternatives could be developed.
    """
)
st.markdown("--- ")
