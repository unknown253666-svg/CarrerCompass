import re

TECH = {"spark","kafka","pyspark","databricks","nlp","vision","ml","pandas","numpy","python","sql","tableau","powerbi","devops","docker","kubernetes","aws","azure","gcp"}
EDU  = {"btech","b.e","be","bachelor","engineering"}
EXP  = re.compile(r"\b(\d+)\+?\s*years?\b")

def boosted_hard_score(resume: str, jd: str, old: float) -> float:
    r, j = resume.lower(), jd.lower()
    tool = 100.0 if {t for t in TECH if t in j} & {t for t in TECH if t in r} else 0.0
    edu  = 100.0 if any(e in r for e in EDU) else 0.0
    exp  = 100.0 if EXP.search(r) else 0.0
    bonus = tool*0.5 + edu*0.3 + exp*0.2
    return round(old*0.6 + bonus*0.4, 2)