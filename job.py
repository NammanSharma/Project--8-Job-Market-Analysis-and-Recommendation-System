import streamlit as st
import pandas as pd
from job_market_analysis import load_and_clean_data, analyze_job_market_trends, create_predictive_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

model = None
vectorizer = None

def get_model_and_vectorizer(df):
    
    global model, vectorizer
    if model is None or vectorizer is None:
        st.write("Training the predictive model... Please wait.")
        hourly_df = df.dropna(subset=['hourly_high']).copy()
        if hourly_df.empty:
            st.error("No hourly rate data available for the predictive model.")
            return None, None
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(hourly_df['title'])
        y = hourly_df['hourly_high']
        model = LinearRegression()
        model.fit(X, y)
    return model, vectorizer

# Set up the main page
st.set_page_config(page_title="Job Market Analysis", layout="wide")
st.title("Job Market Analysis and Recommendation System")
st.write("An interactive application to analyze job trends and predict salaries based on job posting data.")
st.markdown("---")

# Load and clean data (with a spinner for user feedback)
file_path = "all_upwork_jobs_.csv"
with st.spinner("Loading and cleaning data... This may take a moment."):
    jobs_df = load_and_clean_data(file_path)

if jobs_df is not None:
    # --- Section 1: Data Analysis ---
    st.header("1. Job Market Trends Analysis")
    st.write("This section shows key insights derived from the job data.")

    # Task 1: Correlation between Job Title Keywords and Salaries
    st.subheader("Correlation between Job Title Keywords and Salaries")
    # This part of the code is adapted from job_market_analysis.py
    keyword_categories = {
        'Developer': ['developer', 'engineer', 'programmer', 'full stack'],
        'Analyst': ['analyst', 'data scientist', 'machine learning', 'ai'],
        'Manager': ['manager', 'director', 'lead'],
        'Marketing': ['marketing', 'seo', 'social media', 'advertising'],
        'Writer': ['writer', 'copywriter', 'editor']
    }
    salary_data = []
    for category, keywords in keyword_categories.items():
        mask = jobs_df['title'].str.contains('|'.join(keywords), case=False, na=False)
        category_df = jobs_df[mask]
        if not category_df.empty:
            avg_salary = category_df['salary'].mean()
            salary_data.append({'Category': category, 'Average Salary': avg_salary})

    salary_df = pd.DataFrame(salary_data).sort_values(by='Average Salary', ascending=False)
    st.dataframe(salary_df, use_container_width=True)

    # Task 2: Identify Emerging Job Categories based on Posting Frequency
    st.subheader("Emerging Job Categories by Posting Frequency")
    st.write("This chart shows the trend of job postings for different categories over time.")
    jobs_df['month_year'] = jobs_df['published_date'].dt.to_period('M')
    trend_data = {}
    for category, keywords in keyword_categories.items():
        mask = jobs_df['title'].str.contains('|'.join(keywords), case=False, na=False)
        category_df = jobs_df[mask]
        trend_counts = category_df.groupby('month_year')['title'].count().reset_index()
        trend_counts.rename(columns={'title': category}, inplace=True)
        trend_data[category] = trend_counts
    
    if trend_data:
        merged_trends = pd.DataFrame()
        for key, value in trend_data.items():
            if merged_trends.empty:
                merged_trends = value
            else:
                merged_trends = pd.merge(merged_trends, value, on='month_year', how='outer')
        merged_trends.fillna(0, inplace=True)
        merged_trends['month_year'] = merged_trends['month_year'].astype(str)
        merged_trends.set_index('month_year', inplace=True)
        st.line_chart(merged_trends)

    st.markdown("---")

    # --- Section 2: Predictive Model ---
    st.header("2. Predictive Salary Model")
    st.write("Enter a job title to get a predicted hourly rate.")
    
    model, vectorizer = get_model_and_vectorizer(jobs_df)

    if model and vectorizer:
        user_title = st.text_input("Enter a job title:", "Full Stack Developer required for a startup")
        
        if st.button("Predict Salary"):
            if user_title:
                sample_vector = vectorizer.transform([user_title])
                predicted_rate = model.predict(sample_vector)[0]
                st.success(f"The predicted high hourly rate for '{user_title}' is **${predicted_rate:.2f}**.")
            else:
                st.warning("Please enter a job title to get a prediction.")

else:
    st.error("Failed to load data. Please check the file path.")