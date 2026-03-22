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
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# 1. Custom CSS for Retro Synthwave Aesthetics
# -------------------------------------------------
def apply_custom_css():
    st.markdown("""
        <style>
        /* 🌌 Retro '90s Synthwave Background */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #ffffff;
        }
        
        /* 🖱️ Cyberpunk Crosshair Cursor */
        html, body, [class*="css"] {
            cursor: crosshair !important;
        }

        /* 🎛️ Glowing Input Boxes */
        div[data-baseweb="input"] {
            border: 1px solid #ff00ff !important;
            box-shadow: 0 0 10px rgba(255, 0, 255, 0.4) !important;
            border-radius: 8px !important;
            background-color: rgba(0, 0, 0, 0.2) !important;
        }

        /* 💖 Neon Pink & Cyan Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #ff00ff, #00ffff) !important;
            color: #000000 !important;
            font-weight: 900 !important;
            text-transform: uppercase;
            border: none !important;
            box-shadow: 0 0 15px rgba(255, 0, 255, 0.6) !important;
            transition: all 0.3s ease !important;
            border-radius: 8px !important;
            letter-spacing: 1px;
        }
        .stButton>button:hover {
            transform: scale(1.03);
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.8) !important;
        }

        /* 💎 Vibrant Glass Cards */
        .glass-card, .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border-left: 4px solid #00ffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }

        /* 📝 Neon Typography */
        h1 { color: #00ffff !important; text-shadow: 0 0 15px rgba(0, 255, 255, 0.6); }
        h2, h3 { color: #ff00ff !important; text-shadow: 0 0 10px rgba(255, 0, 255, 0.4); }
        
        /* 🧠 Expander Styling (for the Brain Scan) */
        [data-testid="stExpander"] {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid #00ffff !important;
            border-radius: 8px !important;
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
        
        # Upgraded to N-Grams (reads up to 2 words together) and Naive Bayes!
        self.spam_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        X_spam = self.spam_vectorizer.fit_transform(legit_resumes + spam_resumes)
        y_spam = [0]*len(legit_resumes) + [1]*len(spam_resumes)
        self.spam_model = MultinomialNB().fit(X_spam, y_spam)

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

        # --- UPSKILLING MAP & PROJECTS ---
        self.career_paths = {
            "Software Engineer": {
                "next_role": "Senior Systems Architect", 
                "skills": ["System Design", "Microservices", "Advanced Cloud Architecture (AWS/GCP)"],
                "projects": ["Build a scalable microservices API with Docker", "Develop a real-time chat app using WebSockets", "Create a distributed task queue system"]
            },
            "Data Scientist": {
                "next_role": "Chief Data Officer", 
                "skills": ["AI Strategy", "Data Governance", "MLOps"],
                "projects": ["Deploy an end-to-end ML pipeline on AWS", "Build a real-time fraud detection model", "Create an interactive NLP dashboard using Streamlit"]
            },
            "Product Manager": {
                "next_role": "VP of Product", 
                "skills": ["Portfolio Strategy", "P&L Management", "Market Expansion"],
                "projects": ["Write a PRD for a new SaaS feature", "Conduct a competitive market analysis report", "Design a wireframe-to-launch roadmap"]
            },
            "UI/UX Designer": {
                "next_role": "Creative Director", 
                "skills": ["Design Systems Leadership", "Brand Strategy"],
                "projects": ["Design a complete mobile app wireframe in Figma", "Create an interactive high-fidelity prototype", "Publish a UX case study"]
            },
            "Marketing": {
                "next_role": "Chief Marketing Officer", 
                "skills": ["Budget Management", "Cross-functional Leadership"],
                "projects": ["Run a simulated $10k Google Ads campaign", "Develop a 6-month SEO content strategy", "Build an automated email drip sequence"]
            },
            "Sales Manager": {
                "next_role": "VP of Sales", 
                "skills": ["Enterprise Sales Strategy", "Global Market Expansion"],
                "projects": ["Create a B2B cold outreach playbook", "Build a simulated Salesforce CRM pipeline", "Draft a mock enterprise pitch deck"]
            },
            "HR Generalist": {
                "next_role": "Chief HR Officer", 
                "skills": ["Organizational Design", "Executive Coaching"],
                "projects": ["Design a 30-60-90 day onboarding plan", "Create a modern employee retention strategy", "Draft a remote work policy"]
            },
            "Financial Analyst": {
                "next_role": "Chief Financial Officer", 
                "skills": ["Mergers & Acquisitions", "Investor Relations"],
                "projects": ["Build a 3-statement financial model in Excel", "Create a tech startup valuation report", "Develop a quarterly revenue dashboard"]
            }
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
        # --- 1. SPAM DETECTION ---
        spam_vec = self.spam_vectorizer.transform([resume_text])
        spam_prob = self.spam_model.predict_proba(spam_vec)[0][1]
        
        st.sidebar.warning(f"DEBUG: Spam Score is {spam_prob * 100:.2f}%")
        
        if spam_prob > 0.70:
            return {"status": "Rejected", "reason": f"Spam detected (Confidence: {spam_prob*100:.1f}%)"}
        
        # --- 2. JOB MATCHING ---
        resume_vec = self.job_vectorizer.transform([resume_text])
        similarities = cosine_similarity(resume_vec, self.job_matrix)[0]
        
        all_scores = {self.job_categories[i]: round(similarities[i] * 100, 2) for i in range(len(self.job_categories))}
        
        best_match_idx = similarities.argmax()
        best_role = self.job_categories[best_match_idx]
        match_score = all_scores[best_role]
        
        if match_score == 0: 
            best_role = "Uncategorized / General"
            
        recommended_jobs = random.sample([j for j in self.jobs_db if j['Role'] == best_role], min(len([j for j in self.jobs_db if j['Role'] == best_role]), 4))

        # --- 3. 🧠 AI BRAIN SCAN EXTRACTION ---
        feature_names = self.job_vectorizer.get_feature_names_out()
        triggered_indices = resume_vec.nonzero()[1]
        
        ai_brain_scan = {}
        for idx in triggered_indices:
            word = feature_names[idx]
            weight = resume_vec[0, idx]
            ai_brain_scan[word] = round(weight * 100, 2)
            
        sorted_brain_scan = dict(sorted(ai_brain_scan.items(), key=lambda item: item[1], reverse=True))

        # --- 4. FINAL RETURN ---
        return {
            "status": "Accepted",
            "matched_role": best_role,
            "match_confidence": match_score,
            "all_scores": all_scores,
            "upskilling": self.career_paths.get(best_role, {"next_role": "N/A", "skills": [], "projects": []}),
            "recommended_jobs": recommended_jobs,
            "brain_scan": sorted_brain_scan
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

# --- PAGE 1: LOGIN ---
if st.session_state.page == "Login":
    st.markdown("<h1 style='text-align: center; color: #00ffff;'>Nexus AI Resume Screener</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Sign in to unlock your career potential</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🌐 Enter the Nexus", use_container_width=True):
            with st.spinner("Connecting to Mainframe..."):
                time.sleep(1) 
                st.session_state.page = "Upload"
                st.rerun()

# --- PAGE 2: RESUME INPUT ---
elif st.session_state.page == "Upload":
    st.title("📄 Upload Your Data")
    st.write("Provide your resume so our AI can analyze your career trajectory.")
    
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

    if st.button("🚀 Execute Analysis", type="primary"):
        if resume_text.strip():
            with st.spinner("Processing neural weights..."):
                st.session_state.results = analyzer.analyze(resume_text)
                st.session_state.page = "Dashboard"
                st.rerun()
        else:
            st.error("Please provide a resume first.")

# --- PAGE 3: INTERACTIVE DASHBOARD ---
elif st.session_state.page == "Dashboard":
    res = st.session_state.results
    
    if st.button("← Reboot Analysis"):
        st.session_state.page = "Upload"
        st.rerun()

    if res["status"] == "Rejected":
        st.error("🚨 SPAM DETECTED")
        st.write(res["reason"])
    else:
        st.markdown(f"<h1 style='color: #00ffff;'>Welcome to your Career Dashboard</h1>", unsafe_allow_html=True)
        
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
            df_scores = pd.DataFrame(list(res["all_scores"].items()), columns=['Role', 'Match Percentage'])
            df_scores = df_scores.sort_values(by="Match Percentage", ascending=True)
            
            fig = px.bar(df_scores, x='Match Percentage', y='Role', orientation='h', 
                         color='Match Percentage', color_continuous_scale='Sunsetdark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            st.markdown("### 🕸️ Skill Alignment Radar")
            categories = list(res["all_scores"].keys())
            values = list(res["all_scores"].values())
            
            fig2 = go.Figure(data=go.Scatterpolar(
              r=values + [values[0]], 
              theta=categories + [categories[0]],
              fill='toself',
              line_color='#ff00ff'
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
                    <h4 style='margin:0; color:#00ffff;'>{job['Company']}</h4>
                    <p style='margin:0;'>📍 {job['Location']} | 📅 Hiring: {job['Year']}</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col_skills:
            st.markdown("### 🧗‍♂️ Career Progression Path")
            st.markdown(f"To reach **{res['upskilling']['next_role']}**, acquire these skills:")
            for skill in res['upskilling']['skills']:
                st.markdown(f"""
                <div style='background: rgba(255, 0, 255, 0.1); border-left: 4px solid #ff00ff; padding: 10px; margin-bottom: 10px;'>
                    <b>{skill}</b>
                </div>
                """, unsafe_allow_html=True)
            
            # --- NEW: ADDING SALARY & ATS GAP ANALYSIS TO FILL THE SPACE ---
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 1. Salary Insights
            salaries = {
                "Software Engineer": "$110,000 - $160,000",
                "Data Scientist": "$120,000 - $170,000",
                "Product Manager": "$130,000 - $180,000",
                "UI/UX Designer": "$90,000 - $140,000",
                "Marketing": "$80,000 - $130,000",
                "Sales Manager": "$90,000 - $150,000+ (OTE)",
                "HR Generalist": "$70,000 - $100,000",
                "Financial Analyst": "$85,000 - $125,000"
            }
            est_salary = salaries.get(res['matched_role'], "Market Competitive")
            
            st.markdown("### 💰 Expected Market Salary")
            st.markdown(f"""
            <div class='glass-card' style='border-left: 4px solid #00ffcc; padding: 15px; margin-bottom: 20px;'>
                <h2 style='margin: 0; color: #00ffcc; text-align: center; text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);'>{est_salary}</h2>
            </div>
            """, unsafe_allow_html=True)

            # 2. ATS Keyword Gap Analysis (Dynamic calculation)
            st.markdown("### ⚠️ ATS Keyword Gaps")
            st.write("Add these missing keywords to your resume to increase your match score:")
            
            # Calculate missing words: Target words minus found words
            target_words = set(analyzer.job_skills.get(res['matched_role'], "").split())
            found_words = set(res.get("brain_scan", {}).keys())
            missing_words = list(target_words - found_words)[:6] # Grab up to 6 missing words
            
            if missing_words:
                tags_html = ""
                for word in missing_words:
                    # Creating cool red "warning" pill-shaped tags for missing words
                    tags_html += f"<span style='background: rgba(255, 0, 85, 0.2); border: 1px solid #ff0055; color: white; padding: 5px 12px; border-radius: 20px; margin: 0px 5px 5px 0px; display: inline-block; font-size: 0.9em;'>{word.capitalize()}</span>"
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #00ffcc;'>✅ Excellent! Your resume hit all major keywords for this role.</span>", unsafe_allow_html=True)
                
        # 🛠️ RECOMMENDED PROJECTS
        st.markdown("---")
        st.markdown("### 🛠️ Recommended Portfolio Projects")
        st.write(f"Build these 3 projects to guarantee your success as a **{res['matched_role']}**:")
        
        projects = res['upskilling'].get('projects', [])
        if projects:
            proj_cols = st.columns(len(projects))
            for idx, proj in enumerate(projects):
                with proj_cols[idx]:
                    st.markdown(f"""
                    <div class='glass-card' style='border-top: 4px solid #ff00ff; text-align: center; height: 100%;'>
                        <h4 style='color: white; margin-top: 10px;'>💡 Project {idx + 1}</h4>
                        <p style='color: #00ffff; font-weight: bold;'>{proj}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No specific projects available for this category yet.")

        # 🧠 AI BRAIN SCAN 
        st.markdown("---")
        with st.expander("🧠 AI Brain Scan (Expose the Machine's Logic)", expanded=False):
            st.markdown("### How the AI analyzed the text:")
            st.write("This table exposes the exact vocabulary the Vectorizer recognized, along with the mathematical weight applied to each word.")
            
            if "brain_scan" in res and res["brain_scan"]:
                # Display raw DataFrame (No Matplotlib required)
                df_words = pd.DataFrame(list(res["brain_scan"].items()), columns=["Keyword", "Weight Score"])
                st.dataframe(df_words, use_container_width=True, height=300)
            else:
                st.write("No recognized keywords found.")

        # 📺 PREPARATION HUB (Dynamic YouTube Resources)
        st.markdown("---")
        st.markdown(f"### 🎥 Master Your Next Move: {res['matched_role']} Edition")
        st.write("Hand-picked resources tailored to your specific career trajectory.")

        # Dictionary mapping roles to specific video URLs and descriptions
        # (You can swap these YouTube URLs with your own preferred videos)
        role_videos = {
            "Software Engineer": {
                "resume": {"url": "https://www.youtube.com/watch?v=BYUy1yvjHxE", "title": "Tech Resume Mastery", "desc": "Optimize GitHub links & ATS keywords for SWE."},
                "interview": {"url": "https://www.youtube.com/watch?v=kp3ebBEc-mA", "title": "System Design & LeetCode", "desc": "Master technical rounds and whiteboarding."},
                "fear": {"url": "https://www.youtube.com/watch?v=JmO0bKQikEU", "title": "Imposter Syndrome", "desc": "Beat the 'not good enough' feeling in tech."}
            },
            "Data Scientist": {
                "resume": {"url": "https://www.youtube.com/watch?v=xZ_zGVg_fAA", "title": "Data Portfolio Guide", "desc": "Showcase ML models and Kaggle effectively."},
                "interview": {"url": "https://www.youtube.com/watch?v=nB5wz1W1aZk", "title": "Stats & ML Interviews", "desc": "How to explain complex algorithms simply."},
                "fear": {"url": "https://www.youtube.com/watch?v=Zf7l2SStS9A", "title": "Performance Anxiety", "desc": "Stay calm during live coding and stats tests."}
            },
            "Product Manager": {
                "resume": {"url": "https://www.youtube.com/watch?v=3R1P44aZzZg", "title": "PM Resume Impact", "desc": "Focus on product metrics, OKRs, and impact."},
                "interview": {"url": "https://www.youtube.com/watch?v=0FwE4z9P6B0", "title": "Product Case Studies", "desc": "Ace the product sense and execution rounds."},
                "fear": {"url": "https://www.youtube.com/watch?v=HG68Ymazo18", "title": "Executive Presence", "desc": "Project confidence when presenting to stakeholders."}
            },
            "UI/UX Designer": {
                "resume": {"url": "https://www.youtube.com/watch?v=5Q_xZ_zGVg_fAA", "title": "Design Portfolio", "desc": "Structure your case studies and Figma links."},
                "interview": {"url": "https://www.youtube.com/watch?v=7M_kp3ebBEc", "title": "Whiteboard Challenge", "desc": "Navigate live design exercises with ease."},
                "fear": {"url": "https://www.youtube.com/watch?v=Tt08KmFfIYQ", "title": "Defending Your Design", "desc": "Handle critiques without taking it personally."}
            }
        }

        # Fallback dictionary if the role isn't explicitly defined above
        # Fallback dictionary if the role isn't explicitly defined above
        default_videos = {
            "resume": {"url": "https://www.youtube.com/watch?v=dQJShKPltOk", "title": "Resume Mastery", "desc": "Optimize keywords and layout for ATS systems."},
            "interview": {"url": "https://www.youtube.com/watch?v=AQRondpxmwk", "title": "Interview Confidence", "desc": "Stop rambling and connect with your interviewer."},
            "fear": {"url": "https://www.youtube.com/watch?v=wNuKyowAH0E", "title": "Overcoming Fear", "desc": "Actionable steps to reduce interview anxiety."}
        }

        # Fetch the right videos based on the user's matched role
        vids = role_videos.get(res['matched_role'], default_videos)

        # Render the UI
        yt_col1, yt_col2, yt_col3 = st.columns(3)

        with yt_col1:
            st.markdown(f"#### 📄 {vids['resume']['title']}")
            st.video(vids['resume']['url'])
            st.markdown(f"""
                <div style='background: rgba(0, 255, 255, 0.1); padding: 10px; border-radius: 5px; font-size: 0.9em; border-left: 3px solid #00ffff;'>
                <b>Goal:</b> {vids['resume']['desc']}
                </div>
            """, unsafe_allow_html=True)

        with yt_col2:
            st.markdown(f"#### 🤝 {vids['interview']['title']}")
            st.video(vids['interview']['url'])
            st.markdown(f"""
                <div style='background: rgba(0, 255, 255, 0.1); padding: 10px; border-radius: 5px; font-size: 0.9em; border-left: 3px solid #ff00ff;'>
                <b>Goal:</b> {vids['interview']['desc']}
                </div>
            """, unsafe_allow_html=True)

        with yt_col3:
            st.markdown(f"#### 🧘 {vids['fear']['title']}")
            st.video(vids['fear']['url'])
            st.markdown(f"""
                <div style='background: rgba(0, 255, 255, 0.1); padding: 10px; border-radius: 5px; font-size: 0.9em; border-left: 3px solid #00ffff;'>
                <b>Goal:</b> {vids['fear']['desc']}
                </div>
            """, unsafe_allow_html=True)