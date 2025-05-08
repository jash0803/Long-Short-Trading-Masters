import streamlit as st

st.set_page_config(page_title="Investment Questionnaire", page_icon="📝", layout="centered")

st.title("📝 Investment Knowledge and Financial Profile Questionnaire")
st.markdown("---")

# Demographics Section
st.header("👤 Demographics")
st.markdown("Please provide your demographic details:")

age_group = st.radio("**1. What age group do you belong to?**", [
    "Younger than 35 years old",
    "Between 36 and 50",
    "Between 51 and 65",
    "Between 66 and 80",
    "Older than 80 years old"
])

gender = st.radio("**2. Gender:**", [
    "Woman",
    "Man"
])

marital_status = st.radio("**3. You are:**", [
    "Single",
    "Married",
    "Divorced",
    "Widower"
])

education = st.radio("**4. Regarding your studies, you have completed:**", [
    "Elementary Education",
    "Secondary Education",
    "Studies in a Higher Educational Institution / Higher Technological Educational Institution / Other educational institution",
    "Postgraduate studies",
    "Doctoral studies"
])

professional_status = st.radio("**5. What is your current professional status?**", [
    "I do not work / I am unemployed",
    "I am a freelancer / entrepreneur",
    "I work in the private sector",
    "I work in the public sector",
    "I am retired"
])

st.markdown("---")

# Investment Knowledge and Experience Section
st.header("📈 Investment Knowledge and Experience")
st.markdown("Tell us about your knowledge and experience in investments:")

investment_knowledge = st.radio("**6. How would you describe your level of investment knowledge?**", [
    "Low - Not interested and no professional/academic relation to investments",
    "Average - Occasionally update on main financial news",
    "Important - Regularly follow news or professionally related to investments",
    "High - Constantly informed via books, internet, seminars, and professionally involved"
])

investment_experience = st.radio("**7. What is your investment experience?**", [
    "No or minimal experience (Fixed deposits, Bonds, Cash Accounts)",
    "Moderate experience (Bonds, Short-term Guaranteed Capital Products)",
    "Significant experience (Shares, Equity Accounts, Long-term Guaranteed Products)",
    "Extensive experience (Derivatives, Structured Products)"
])

trading_frequency = st.radio("**8. How often did you make trades in the last three years?**", [
    "Rarely (1-2 times a year)",
    "Occasionally (every 2-3 months)",
    "Often (every fortnight or month)",
    "Very often (at least 2 times a week)"
])

st.markdown("---")

# Financial Independence and Liquidity Section
st.header("💰 Financial Independence and Liquidity")
st.markdown("Please tell us about your income and financial stability:")

annual_income = st.radio("**9. What is your annual income?**", [
    "Under 30,000 euros",
    "30,001 - 60,000 euros",
    "60,001 - 90,000 euros",
    "90,001 - 120,000 euros",
    "Above 120,000 euros"
])

income_source = st.radio("**10. Your regular income comes mainly from:**", [
    "Agricultural work",
    "Salaried services - pensions",
    "Liberal professions",
    "Commercial enterprises"
])

st.markdown("---")

# Submit Button
if st.button("Submit Questionnaire"):
    st.success("🎉 Thank you for submitting your responses!")

