import os, google.generativeai as genai
from typing import List

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_missing_skills(resume: str, jd: str) -> List[str]:
    resume_lower = resume.lower()
    prompt = "From the job description below, list ONLY concrete technical skills, tools, libraries, frameworks, cloud services, databases, languages. Return comma-separated, nothing else.\n\nJob Description:\n{jd}\n\nSkills:"
    reply = model.generate_content(prompt.format(jd=jd), generation_config={"temperature": 0.0}).text
    skills = [s.strip().lower() for s in reply.split(",") if s.strip()]
    return [s for s in skills if s not in resume_lower and len(s) > 2][:12]