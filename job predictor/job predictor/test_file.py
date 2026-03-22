# ============================================
# AESTHETIC AI RESUME SCREENER (WEB APP)
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import csv
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# -------------------------------------------------
# 1. Custom CSS for Aesthetics, Animations & Cursors
# -------------------------------------------------
def apply_custom_css():
    st.markdown("""
        <style>
        /* Custom Cursor */
        html, body, [class*="css"] {
            cursor: crosshair !important;
        }
        
        /* Animated Gradient Background for Main Container */
        .stApp {
            background: linear-gradient(-45deg, #0f172a, #1e293b, #0f172a, #334155);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: white;
        }
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* Google Login Button Styling */
        .google-btn {
            background-color: white;
            color: #444;
            border-radius: 5px;
            border: thin solid #888;
            box-shadow: 1px 1px 1px grey;
            white-space: nowrap;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: 0.3s;
        }
        .google-btn:hover {
            box-shadow: 2px 2px 5px #888;
            transform: scale(1.02);
        }
        
        /* Card styling */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# 2. Machine Learning Pipeline
# -------------------------------------------------
class ResumeAnalyzerML:
    def __init__(self):
        self.csv_filename = "jobs_database.csv"
        self.jobs_db = []
        self.locations = ["Bengaluru", "Chennai", "Hyderabad", "Pune", "Mumbai", "Gurugram", "Noida"]

        # --- SPAM DETECTION ---
        legit_resumes = [
            "Software engineer with 4 years of experience in Python, Java, C++, and SQL.",
            "Backend developer optimizing high-latency microservices with C++ and Go.",
            "Digital marketing manager with strong expertise in SEO, branding, content strategy.",
            "Finance analyst with experience in financial reporting, accounting, taxation.",
            "Human resources executive experienced in recruitment, onboarding, payroll.",
            "Operations manager with background in supply chain management, logistics.",
            "Data scientist building predictive models using Python, TensorFlow.",
            "UI/UX designer creating wireframes, prototypes, and user research using Figma.",
            "Product manager leading cross-functional teams, agile scrum.",
            "Cloud architect designing scalable infrastructure using AWS and Kubernetes.",
            "Cybersecurity analyst monitoring network traffic for potential vulnerabilities.",
            "Frontend engineer specializing in React, TypeScript, and responsive web design.",
            "Data analyst proficient in SQL, Tableau, and creating interactive dashboards.",
            "Investment banker with expertise in mergers, acquisitions, and financial modeling.",
            "Social media strategist managing campaigns across platforms to drive engagement.",
            "B2B sales manager with a proven track record of exceeding revenue targets.",
            "Talent acquisition specialist sourcing passive candidates through LinkedIn Recruiter.",
            "Supply chain coordinator ensuring timely delivery and inventory optimization.",
            "Legal counsel drafting corporate contracts and ensuring regulatory compliance.",
            "Quality assurance tester writing automated scripts using Selenium and PyTest.",
            "Healthcare administrator managing clinic operations and patient scheduling.",
            "Mechanical engineer utilizing AutoCAD and SolidWorks for product design.",
            "Public relations specialist drafting press releases and managing media relations.",
            "Network engineer configuring routers, switches, and firewalls for enterprise networks.",
            "Technical writer creating clear API documentation and user manuals.",
            "Executive assistant managing complex calendars and international travel arrangements.",
            "Machine learning engineer deploying NLP models into production environments.",
            "Graphic designer creating branding materials using Adobe Creative Cloud.",
            "Customer success manager focused on reducing churn and increasing user adoption.",
            "Scrum master facilitating sprint planning, daily stand-ups, and retrospectives."
        ]
        
        spam_resumes = [
            "Earn money fast click here free job guaranteed no experience needed.",
            "Congratulations you won lottery apply now urgent job wire transfer.",
            "Make 5000 daily without skills work from home easy cash.",
            "Free certificate guaranteed high salary no experience required.",
            "Click link now limited offer job instantly no interview needed.",
            "Work from home envelope stuffing easy cash fast payment no skills.",
            "Be your own boss financial freedom multi level marketing pyramid downline.",
            "Deposit this check keep 20 percent and wire the rest to our agent.",
            "Claim your prize money now provide bank account details for direct deposit.",
            "Make six figures from your couch with zero effort easy money fast.",
            "Start earning 500 dollars an hour today no qualifications necessary.",
            "Earn passive income instantly sign up now and get a free cash bonus.",
            "Unlock the secret to wealth buy my course and become a millionaire overnight.",
            "Verify your identity by clicking this secure link to claim your dream job.",
            "Guaranteed job placement just send your social security number and credit card.",
            "Congratulations your email was selected for a million dollar cash prize.",
            "Instant payout western union wire transfer easy cash no background check.",
            "No resume needed no interview required just click the link to start earning.",
            "Urgent hiring for data entry no interview needed just pay processing fee.",
            "Act now limited time offer urgent vacancy instant hire cash advance.",
            "Work from home crypto investment guaranteed returns double your income.",
            "Risk free investment guaranteed profit work entirely online easy steps.",
            "Become a secret shopper keep the merchandise and get paid via money order.",
            "You have been selected for an exclusive VIP offer click here to claim reward.",
            "Huge salary for simple copy paste jobs guaranteed payment today.",
            "Avoid the 9 to 5 trap click this link to discover the ultimate money glitch.",
            "Get paid to watch videos urgent hire instant approval direct deposit.",
            "Earn big rewards taking surveys online fast cash payout guaranteed.",
            "Need extra cash sign up today and we will wire you a sign on bonus.",
            "Limited slots available for secret money making system apply immediately."
        ]
        
        self.spam_vectorizer = TfidfVectorizer(stop_words='english')
        X_spam = self.spam_vectorizer.fit_transform(legit_resumes + spam_resumes)
        y_spam = [0]*len(legit_resumes) + [1]*len(spam_resumes)
        self.spam_model = LogisticRegression().fit(X_spam, y_spam)

        # --- JOB MATCHING ---
        self.job_skills = {
            "Software Engineer": "python java c++ programming data structures algorithms backend cloud sql database architecture",
            "Data Scientist": "python r sql machine learning deep learning nlp tensorflow pandas data analysis statistics",
            "Product Manager": "agile scrum roadmap strategy user experience jira analytics a/b testing cross-functional",
            "UI/UX Designer": "figma adobe xd wireframing prototyping user research visual design ui ux interface",
            "Marketing": "seo branding digital marketing analytics content strategy social media campaigns ads",
            "Sales Manager": "b2b b2c crm salesforce lead generation negotiation account management pipeline revenue",
            "HR Generalist": "recruitment onboarding payroll employee engagement talent acquisition benefits compliance",
            "Financial Analyst": "accounting taxation financial modeling risk management excel forecasting budget reporting"
        }
        self.job_categories = list(self.job_skills.keys())
        self.job_descriptions = list(self.job_skills.values())
        self.job_vectorizer = TfidfVectorizer(stop_words='english')
        self.job_matrix = self.job_vectorizer.fit_transform(self.job_descriptions)

        # --- UPSKILLING MAP ---
        self.career_paths = {
            "Software Engineer": {"next_role": "Senior Systems Architect", "skills": ["System Design", "Microservices", "Advanced Cloud Architecture (AWS/GCP)"]},
            "Data Scientist": {"next_role": "Chief Data Officer", "skills": ["AI Strategy", "Data Governance", "MLOps"]},
            "Product Manager": {"next_role": "VP of Product", "skills": ["Portfolio Strategy", "P&L Management", "Market Expansion"]},
            "UI/UX Designer": {"next_role": "Creative Director", "skills": ["Design Systems Leadership", "Brand Strategy"]},
            "Marketing": {"next_role": "Chief Marketing Officer", "skills": ["Budget Management", "Cross-functional Leadership"]},
            "Sales Manager": {"next_role": "VP of Sales", "skills": ["Enterprise Sales Strategy", "Global Market Expansion"]},
            "HR Generalist": {"next_role": "Chief HR Officer", "skills": ["Organizational Design", "Executive Coaching"]},
            "Financial Analyst": {"next_role": "Chief Financial Officer", "skills": ["Mergers & Acquisitions", "Investor Relations"]}
        }
        self._load_jobs_database()

    def _load_jobs_database(self):
        if not os.path.exists(self.csv_filename):
            companies = ["TCS", "Infosys", "Wipro", "Zoho", "Google", "Amazon", "Flipkart", "Paytm"]
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Role", "Company", "Location", "Year"])
                for role in self.job_categories:
                    for _ in range(5):
                        writer.writerow([role, random.choice(companies), random.choice(self.locations), random.choice(["2025", "2026"])])
        
        with open(self.csv_filename, 'r', encoding='utf-8') as file:
            self.jobs_db = list(csv.DictReader(file))

    def analyze(self, resume_text):
        spam_vec = self.spam_vectorizer.transform([resume_text])
        spam_prob = self.spam_model.predict_proba(spam_vec)[0][1]
        
        if spam_prob > 0.75:
            return {"status": "Rejected", "reason": f"Spam detected (Confidence: {spam_prob*100:.1f}%)"}
        
        resume_vec = self.job_vectorizer.transform([resume_text])
        similarities = cosine_similarity(resume_vec, self.job_matrix)[0]
        
        # Capture ALL scores for the bar chart
        all_scores = {self.job_categories[i]: round(similarities[i] * 100, 2) for i in range(len(self.job_categories))}
        
        best_match_idx = similarities.argmax()
        best_role = self.job_categories[best_match_idx]
        match_score = all_scores[best_role]
        
        if match_score == 0: best_role = "Uncategorized / General"
            
        recommended_jobs = random.sample([j for j in self.jobs_db if j['Role'] == best_role], min(len([j for j in self.jobs_db if j['Role'] == best_role]), 4))
        
        return {
            "status": "Accepted",
            "matched_role": best_role,
            "match_confidence": match_score,
            "all_scores": all_scores,
            "upskilling": self.career_paths.get(best_role, {"next_role": "N/A", "skills": []}),
            "recommended_jobs": recommended_jobs
        }

# -------------------------------------------------
# 3. Streamlit Page Routing & UI
# -------------------------------------------------
st.set_page_config(page_title="Nexus AI Career", page_icon="✨", layout="wide")
apply_custom_css()

@st.cache_resource
def load_model():
    return ResumeAnalyzerML()

analyzer = load_model()

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = "Login"
if 'results' not in st.session_state:
    st.session_state.results = None

# --- PAGE 1: LOGIN & UPLOAD ---
if st.session_state.page == "Login":
    st.markdown("<h1 style='text-align: center; color: #00f2fe;'>Nexus AI Resume Screener</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Sign in to unlock your career potential</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        # Simulated Google Login UI
        if st.button("🌐 Sign in with Google", use_container_width=True):
            with st.spinner("Authenticating..."):
                time.sleep(1) # Simulate network delay
                st.session_state.page = "Upload"
                st.rerun()

# --- PAGE 2: RESUME INPUT ---
elif st.session_state.page == "Upload":
    st.title("📄 Upload Your Resume")
    st.write("Welcome! Please provide your resume so our AI can analyze your career trajectory.")
    
    resume_text = ""
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text = page.extract_text()
            if text: resume_text += text + "\n"
    
    st.markdown("### OR")
    text_input = st.text_area("Paste Resume Text Here:", height=150)
    if text_input: resume_text = text_input

    if st.button("🚀 Analyze Resume", type="primary"):
        if resume_text.strip():
            with st.spinner("AI is analyzing your skills and matching jobs..."):
                st.session_state.results = analyzer.analyze(resume_text)
                st.session_state.page = "Dashboard"
                st.rerun()
        else:
            st.error("Please provide a resume first.")

# --- PAGE 3: INTERACTIVE DASHBOARD ---
elif st.session_state.page == "Dashboard":
    res = st.session_state.results
    
    if st.button("← Back to Upload"):
        st.session_state.page = "Upload"
        st.rerun()

    if res["status"] == "Rejected":
        st.error("🚨 SPAM DETECTED")
        st.write(res["reason"])
    else:
        st.markdown(f"<h1 style='color: #00c6ff;'>Welcome to your Career Dashboard</h1>", unsafe_allow_html=True)
        
        # TOP METRICS
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Primary Match", res["matched_role"])
        c2.metric("📊 Confidence Score", f"{res['match_confidence']}%")
        c3.metric("🚀 Next Target Role", res["upskilling"]["next_role"])
        
        st.markdown("---")
        
        # CHARTS SECTION
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 📈 Role Suitability Distribution")
            # Bar Chart using Plotly
            df_scores = pd.DataFrame(list(res["all_scores"].items()), columns=['Role', 'Match Percentage'])
            df_scores = df_scores.sort_values(by="Match Percentage", ascending=True)
            
            fig = px.bar(df_scores, x='Match Percentage', y='Role', orientation='h', 
                         color='Match Percentage', color_continuous_scale='Tealgrn')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            st.markdown("### 🕸️ Skill Alignment Radar")
            # Radar Chart using Plotly Graph Objects
            categories = list(res["all_scores"].keys())
            values = list(res["all_scores"].values())
            
            fig2 = go.Figure(data=go.Scatterpolar(
              r=values + [values[0]], # Close the line
              theta=categories + [categories[0]],
              fill='toself',
              line_color='#00f2fe'
            ))
            fig2.update_layout(
              polar=dict(radialaxis=dict(visible=True, range=[0, max(values)+10])),
              showlegend=False,
              paper_bgcolor='rgba(0,0,0,0)', font_color="white"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        
        # JOB RECOMMENDATIONS & UPSKILLING
        col_jobs, col_skills = st.columns(2)
        
        with col_jobs:
            st.markdown("### 🏢 Active Job Openings")
            for job in res['recommended_jobs']:
                st.markdown(f"""
                <div class='glass-card'>
                    <h4 style='margin:0; color:#00f2fe;'>{job['Company']}</h4>
                    <p style='margin:0;'>📍 {job['Location']} | 📅 Hiring: {job['Year']}</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col_skills:
            st.markdown("### 🧗‍♂️ Career Progression Path")
            st.markdown(f"To reach **{res['upskilling']['next_role']}**, acquire these skills:")
            for skill in res['upskilling']['skills']:
                st.markdown(f"""
                <div style='background: rgba(0, 242, 254, 0.1); border-left: 4px solid #00f2fe; padding: 10px; margin-bottom: 10px;'>
                    <b>{skill}</b>
                </div>
                """, unsafe_allow_html=True)