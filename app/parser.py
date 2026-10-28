import spacy
from spacy.pipeline import EntityRuler
import re
import logging
import os
import json
from app.normalizer import normalizer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Note: 100% Offline NLP Parser (Google Gemini AI removed for zero-latency, local extraction)


US_WORK_AUTH_PATTERNS = {
    "US Citizen": [
        r"\bu\.?\s*s\.?\s*citizen\b",
        r"\bcitizenship\s*:\s*(u\.?\s*s\.?\s*a?|united\s+states)\b",
        r"\bus\s+citizen\b",
        r"\bvalid\s+usc\b"
    ],
    "Green Card": [
        r"\bgreen\s*card\b",
        r"\bpermanent\s+resident\b",
        r"\bgc\s+holder\b",
        r"\bgc\b(?!\s*-\s*ms)\b(?!\s+analysis\b)"
    ],
    "H1B": [
        r"\bh1\s*b\b",
        r"\bh-1b\b",
        r"\bh1b\s+visa\b"
    ],
    "H4 EAD": [
        r"\bh4\s*ead\b",
        r"\bh-4\s*ead\b"
    ],
    "GC EAD": [
        r"\bgc\s*ead\b"
    ],
    "L2 EAD": [
        r"\bl2\s*ead\b",
        r"\bl-2\s*ead\b"
    ],
    "OPT EAD": [
        r"\bopt\s+ead\b",
        r"\bstem\s+opt\b",
        r"\bf1\s+opt\b",
        r"\bopt\b(?!\s+(?:optimization|compiler|design|skills))\b"
    ],
    "CPT": [
        r"\bcpt\b",
        r"\bcurricular\s+practical\s+training\b"
    ],
    "TN Visa": [
        r"\btn\s+visa\b",
        r"\btn-1\b",
        r"\btn-2\b"
    ],
    "EAD": [
        r"\bead\b",
        r"\bemployment\s+authorization\s+document\b"
    ]
}

def extract_work_authorization(text: str) -> str:
    """
    Scans the raw resume text using heuristics to detect US work authorization.
    """
    text_lower = text.lower()
    
    # 1. Context-based scan (high precision)
    context_lines = []
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["work permit", "authorization", "visa status", "legal status", "citizenship", "visa:"]):
            context_lines.append(line_lower)
            
    if context_lines:
        context_block = "\n".join(context_lines)
        for canonical, patterns in US_WORK_AUTH_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, context_block):
                    return canonical
                    
    # 2. Document-wide scan (fallback with safe word boundary regexes)
    for canonical, patterns in US_WORK_AUTH_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return canonical
                
    return None


def create_nlp_pipeline():
    """Initializes a spaCy pipeline with custom entity rules for resume parsing."""
    try:
        # Using md model for better entity recognition
        nlp = spacy.load("en_core_web_md")
    except OSError:
        logger.warning("spaCy model 'en_core_web_md' not found. Falling back to 'en_core_web_sm'.")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return None

    # Check if entity_ruler already exists
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")
    
    # 1. Define Patterns for Resume Entities
    patterns = [
        # Section Headers (Used as fallback or for semantic parsing)
        {"label": "SECTION", "pattern": [{"LOWER": {"IN": ["experience", "employment", "work"]}}, {"LOWER": {"IN": ["history", "background", "experience"]}, "OP": "?"}]},
        {"label": "SECTION", "pattern": [{"LOWER": {"IN": ["education", "academic", "scholastic"]}}, {"LOWER": {"IN": ["history", "background", "qualifications", "education"]}, "OP": "?"}]},
        {"label": "SECTION", "pattern": [{"LOWER": {"IN": ["skills", "technologies", "proficiencies", "expertise", "tools"]}}]},
        {"label": "SECTION", "pattern": [{"LOWER": {"IN": ["certifications", "certificates", "training", "courses"]}}]},
        {"label": "SECTION", "pattern": [{"LOWER": {"IN": ["projects", "assignments"]}}]},
        
        # Degrees
        {"label": "DEGREE", "pattern": [{"LOWER": "mba"}]},
        {"label": "DEGREE", "pattern": [{"LOWER": "mca"}]},
        {"label": "DEGREE", "pattern": [{"LOWER": "b.tech"}]},
        {"label": "DEGREE", "pattern": [{"LOWER": "m.tech"}]},
        {"label": "DEGREE", "pattern": [{"LOWER": "post"}, {"LOWER": "graduation"}]},
        {"label": "DEGREE", "pattern": [{"LOWER": "graduation"}]},
        {"label": "DEGREE", "pattern": [{"LOWER": "bachelor"}, {"LOWER": "of"}, {"LOWER": {"IN": ["technology", "science", "arts", "engineering", "commerce"]}}]},
        {"label": "DEGREE", "pattern": [{"LOWER": "master"}, {"LOWER": "of"}, {"LOWER": {"IN": ["technology", "science", "arts", "engineering", "commerce", "business"]}}]},
        {"label": "DEGREE", "pattern": [{"LOWER": {"IN": ["b.sc", "m.sc", "b.e", "m.e", "b.com", "m.com", "phd", "bca"]}}]},
        
        # Cloud & DevOps
        {"label": "SKILL", "pattern": [{"LOWER": {"IN": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ansible", "linux", "git", "github", "gitlab"]}}]},
        # Frontend & Mobile
        {"label": "SKILL", "pattern": [{"LOWER": {"IN": ["react", "angular", "vue", "next.js", "flutter", "react native", "html", "css", "sass", "tailwind", "bootstrap", "typescript"]}}]},
        # Backend & Data
        {"label": "SKILL", "pattern": [{"LOWER": {"IN": ["python", "java", "spring", "node.js", "express", "go", "ruby", "django", "flask", "fastapi", "sql", "postgresql", "mongodb", "redis", "elasticsearch"]}}]},
        # HR & Business
        {"label": "SKILL", "pattern": [{"LOWER": {"IN": ["hris", "hrms", "ats", "payroll", "recruitment", "onboarding", "compliance", "epfo", "esic", "tally", "sap", "erp"]}}]}
    ]
    
    ruler.add_patterns(patterns)
    return nlp

# Global NLP Instance
nlp = create_nlp_pipeline()

def normalize_text_dashes(t: str) -> str:
    """Normalizes unicode dashes, private use icons, and adds space between month names and 4-digit years."""
    if not t:
        return ""
    t = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2212]', '-', t)
    t = re.sub(r'[\uf000-\uf8ff]', '', t)
    # Only separate months and years like June2023 -> June 2023, never email addresses like barik2808
    t = re.sub(r'(?i)\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(\d{4})\b', r'\1 \2', t)
    return t

def split_date_range(raw_str: str):
    raw_str = raw_str.strip()
    parts = re.split(r'\s*[-–—/]\s*|\s+to\s+', raw_str, flags=re.I)
    start = parts[0].strip() if len(parts) > 0 else raw_str
    end = parts[1].strip() if len(parts) > 1 else ""
    if len(start) == 4 and len(end) == 2 and end.isdigit():
        end = start[:2] + end
    return start, end

HEADER_PATTERNS = {
    "experience": r"(?i)^\s*(experience|employment|work\s+history|employment\s+history|professional\s+experience|professional\s+background|work\s+experience|relevant\s+experience|career\s+history|career\s+summary|roles\s*(?:&|and)\s*responsibilities|industry\s+experience)\s*[:]?\s*$",
    "education": r"(?i)^\s*(education|academic|scholastic|qualifications?|academic\s+qualifications?|educational\s+qualifications?|academic\s+history|education\s+history|academic\s+background|educational\s+background|academics|education\s*(?:&|and)\s*qualifications?|education\s*(?:&|and)\s*training|degrees?\s*(?:&|and)\s*education)\s*[:]?\s*$",
    "skills": r"(?i)^\s*(skills|technologies|proficiencies|expertise|technical\s+skills|key\s+skills|core\s+competencies|competencies|tools|programming|technical\s+expertise|areas\s+of\s+expertise|technology\s+stack|technical\s+proficiencies|skills\s*(?:&|and)\s*(?:tools|technologies))\s*[:]?\s*$",
    "certifications": r"(?i)^\s*(certifications?|certificates?|training|courses|courses\s+and\s+certifications|licenses?\s*(?:&|and)\s*certifications?|awards)\s*[:]?\s*$",
    "projects": r"(?i)^\s*(projects|key\s+projects|academic\s+projects|recent\s+projects|personal\s+projects|selected\s+projects|assignments|portfolio)\s*[:]?\s*$",
    "internships": r"(?i)^\s*(internships?|summer\s+internship|vocational\s+training)\s*[:]?\s*$",
    "interests": r"(?i)^\s*(interests?|hobbies|activities|extracurricular(?:\s+activities)?)\s*[:]?\s*$",
    "languages": r"(?i)^\s*(languages?|known\s+languages|languages\s+known)\s*[:]?\s*$",
    "declaration": r"(?i)^\s*(declaration|declaration\s*:?)\s*[:]?\s*$",
    "links": r"(?i)^\s*(links?|social\s+links|profiles?)\s*[:]?\s*$",
    "summary": r"(?i)^\s*(objective|career\s+objective|summary|professional\s+summary|executive\s+summary|profile|about\s+me)\s*[:]?\s*$",
    "additional": r"(?i)^\s*(additional\s+information|personal\s+details|personal\s+information)\s*[:]?\s*$"
}

DATE_RANGE_REGEX = re.compile(
    r'(?:(?:\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\.?\s*)?\d{2,4}\s*(?:[-–—/]+|\bto\b)\s*(?:(?:\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\.?\s*)?\d{2,4}|Present|Current|Till Date|Ongoing)\b)',
    re.I
)
YEAR_RANGE_REGEX = re.compile(r'\b(19|20)\d{2}\s*(?:[-–—/]+|\bto\b)\s*(?:(19|20)?\d{2}|Present|Current)\b', re.I)
SINGLE_YEAR_REGEX = re.compile(r'\b(19|20)\d{2}\b')

DEGREE_REGEX = re.compile(
    r'\b(b\.?tech\.?|m\.?tech\.?|b-tech|m-tech|b\.?e\.?|m\.?e\.?|b\.?sc\.?|m\.?sc\.?|b\.?com\.?|m\.?com\.?|bca|mca|mba|ph\.?d\.?|bachelor(?:\s+of\s+[\w\s&]+)?|master(?:\s+of\s+[\w\s&]+)?|diploma|class\s+(?:xii|x|12|10)|12th|10th|xii|x|higher\s+secondary|secondary\s+school|high\s+school)\b',
    re.I
)

ROLE_REGEX = re.compile(
    r'\b(manager|engineer|consultant|analyst|developer|lead|assistant|executive|specialist|officer|coordinator|head|director|intern|trainee|recruiter|generalist|associate|supervisor|administrator|architect|scientist)\b',
    re.I
)

TABLE_HEADERS = {
    "qualification", "qualifications", "degree", "institution", "institute", 
    "university", "college", "school", "board", "year", "passing year", 
    "year of passing", "percentage", "cgpa", "sgpa", "marks", "grade", 
    "duration", "stream", "specialization", "major"
}

def extract_header_location(text: str):
    """Extracts location from candidate header / contact block."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    header_block = lines[:15]
    known_countries = {"india", "usa", "united states", "uk", "canada", "australia", "singapore", "uae", "germany"}
    
    for line in header_block:
        # Split on pipe, bullet, or tab (preserve commas within location)
        segments = [s.strip() for s in re.split(r'[|•●▪\t;]', line) if s.strip()]
        for seg in segments:
            if "@" in seg or re.search(r'\d{7,}', seg) or any(k in seg.lower() for k in ["http", "www", "github", "linkedin", "gitlab"]):
                continue
            parts = [p.strip() for p in seg.split(',') if p.strip()]
            if 1 <= len(parts) <= 3:
                seg_lower = seg.lower()
                if any(c in seg_lower for c in [
                    "bengaluru", "bangalore", "pune", "mumbai", "delhi", "hyderabad", "chennai", 
                    "kolkata", "noida", "gurgaon", "san francisco", "new york", "austin", "dallas", 
                    "chicago", "seattle", "california", "texas", "karnataka", "maharashtra"
                ]) or any(c in seg_lower for c in known_countries):
                    return {
                        "raw": seg,
                        "city": parts[0],
                        "state": parts[1] if len(parts) > 1 else "",
                        "country": parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 and parts[1].lower() in known_countries else "")
                    }
    return {"raw": "", "city": "", "state": "", "country": ""}

def parse_education_detailed(edu_text: str):
    if not edu_text:
        return []
    edu_text = normalize_text_dashes(edu_text)
    raw_lines = [re.sub(r'^[^\w\s()/,.-]+', '', l).strip() for l in edu_text.split('\n') if l.strip()]
    if not raw_lines:
        return []

    # Filter out table header rows
    filtered = []
    for l in raw_lines:
        cleaned = l.lower().strip()
        parts = [p.strip() for p in re.split(r'[\t|/]+', cleaned) if p.strip()]
        if parts and all(p in TABLE_HEADERS for p in parts):
            continue
        if cleaned in TABLE_HEADERS:
            continue
        filtered.append(l)

    # Group into entries
    entries = []
    curr_entry = []
    for line in filtered:
        is_degree = bool(DEGREE_REGEX.search(line))
        is_year = bool(SINGLE_YEAR_REGEX.search(line) and len(line.split()) <= 2)
        
        curr_has_deg = any(DEGREE_REGEX.search(l) for l in curr_entry)
        curr_has_year = any(SINGLE_YEAR_REGEX.search(l) and len(l.split()) <= 2 for l in curr_entry)
        
        if is_degree and curr_has_deg:
            entries.append(curr_entry)
            curr_entry = []
        elif is_year and curr_has_year:
            entries.append(curr_entry)
            curr_entry = []
            
        curr_entry.append(line)
    if curr_entry:
        entries.append(curr_entry)

    results = []
    for grp in entries:
        degree = ""
        major = ""
        school = ""
        start_date = ""
        end_date = ""
        for line in grp:
            d_match = YEAR_RANGE_REGEX.search(line)
            if d_match and not start_date:
                parts = re.split(r'\s*[-–—/]\s*|\s+to\s+', d_match.group(0), flags=re.I)
                start_date = parts[0].strip()
                end_date = parts[1].strip() if len(parts) > 1 else ""
                continue
            sy_match = SINGLE_YEAR_REGEX.search(line)
            if sy_match and not end_date and len(line.split()) <= 3:
                end_date = sy_match.group(0)
                continue

            deg_match = DEGREE_REGEX.search(line)
            if deg_match and not degree:
                degree = line
                if " - " in line:
                    major = line.split(" - ", 1)[1].strip()
                elif " in " in line.lower():
                    major = re.split(r'\s+in\s+', line, flags=re.I)[1].strip()
                continue

            if any(k in line.lower() for k in ["university", "college", "school", "institute", "vidyalaya", "academy", "polytechnic", "board"]) and not school:
                school = line
                continue
            elif not school and not deg_match and not d_match and not sy_match:
                school = line

        if degree or school:
            results.append({
                "degree": degree or "Degree",
                "school": school,
                "major": major,
                "start_date": start_date,
                "end_date": end_date
            })
    return results

def parse_experience_detailed(exp_text: str):
    if not exp_text:
        return []
    exp_text = normalize_text_dashes(exp_text)
    raw_lines = [l.strip() for l in exp_text.split('\n') if l.strip()]
    if not raw_lines:
        return []

    # Group lines by job entry
    entries = []
    curr_lines = []
    
    for line in raw_lines:
        clean_l = re.sub(r'^[^\w\s()/,.-]+', '', line).strip()
        date_match = DATE_RANGE_REGEX.search(clean_l) or YEAR_RANGE_REGEX.search(clean_l)
        is_role = bool(ROLE_REGEX.search(clean_l) and len(clean_l.split()) <= 4 and not line.startswith(('•', '-', '*', '●')))
        
        has_date = any(DATE_RANGE_REGEX.search(l) or YEAR_RANGE_REGEX.search(l) for l in curr_lines)
        
        if date_match and has_date:
            entries.append(curr_lines)
            curr_lines = []
        elif is_role and has_date and len(curr_lines) >= 3:
            entries.append(curr_lines)
            curr_lines = []
            
        curr_lines.append(line)
        
    if curr_lines:
        entries.append(curr_lines)

    results = []
    for lines in entries:
        role = ""
        company = ""
        start_date = ""
        end_date = ""
        location = ""
        bullets = []
        
        for line in lines:
            clean_l = re.sub(r'^[^\w\s()/,.-]+', '', line).strip()
            
            # Check date line
            d_match = DATE_RANGE_REGEX.search(clean_l) or YEAR_RANGE_REGEX.search(clean_l)
            if d_match:
                raw_date_str = d_match.group(0)
                if not start_date:
                    parts = re.split(r'\s*[-–—/]\s*|\s+to\s+', raw_date_str, flags=re.I)
                    start_date = parts[0].strip()
                    end_date = parts[1].strip() if len(parts) > 1 else ""
                
                # Check if line also contains company / location
                if '|' in line or ' - ' in line.replace(raw_date_str, ''):
                    remainder = line.replace(raw_date_str, '').strip(' \t|-–—')
                    if remainder and not company:
                        comp_parts = [p.strip() for p in remainder.split(',') if p.strip()]
                        if comp_parts:
                            company = comp_parts[0]
                            if len(comp_parts) > 1:
                                location = ", ".join(comp_parts[1:])
                continue
                
            r_match = ROLE_REGEX.search(clean_l)
            if r_match and not role and len(clean_l.split()) <= 6 and not line.startswith(('•', '-', '*', '●')):
                role = clean_l
                continue
                
            # Check company (strictly exclude universities/colleges)
            is_edu = any(k in clean_l.lower() for k in ["university", "college", "school", "institute", "academy"])
            if not is_edu and not company and not line.startswith(('•', '-', '*', '●')):
                is_comp = any(k in clean_l.lower() for k in ["pvt", "ltd", "limited", "services", "consultancy", "solutions", "technologies", "corp", "inc", "llc", "tcs", "infosys", "wipro", "cognizant", "ibm", "accenture"])
                if is_comp or (len(clean_l.split()) >= 2 and len(clean_l.split()) <= 8):
                    comp_parts = [p.strip() for p in clean_l.split(',') if p.strip()]
                    company = comp_parts[0]
                    if len(comp_parts) > 1:
                        location = ", ".join(comp_parts[1:])
                    continue
                    
            if line.startswith(('•', '-', '*', '●')) or len(line.split()) > 7:
                bullets.append(line.lstrip('•-*● '))

        if role or company or start_date:
            results.append({
                "role": role or "Role",
                "company": company or "Company",
                "location": location,
                "start_date": start_date,
                "end_date": end_date or "Present",
                "description": "\n".join(bullets),
                "is_current": "present" in (end_date or "").lower() or "current" in (end_date or "").lower()
            })
    return results

def calculate_experience_years(text: str, experience_detailed: list) -> float:
    """Calculates total experience years either from explicit resume statement or from job dates."""
    # 1. Explicit statement scan
    m = re.search(r'(?i)\b(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)\b', text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 2. Compute date spans from experience_detailed
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    
    def parse_dt(dt_str: str, is_end: bool = False):
        if not dt_str:
            return None
        dt_str = dt_str.strip().lower()
        if "present" in dt_str or "current" in dt_str or "till date" in dt_str:
            return 2026, 9
        ym = re.search(r'\b(19|20\d{2})\b', dt_str)
        if not ym:
            return None
        year = int(ym.group(1))
        month = 12 if is_end else 1
        for m_name, m_num in month_map.items():
            if m_name in dt_str:
                month = m_num
                break
        return year, month

    intervals = []
    for exp in experience_detailed:
        s = parse_dt(exp.get("start_date"), is_end=False)
        e = parse_dt(exp.get("end_date"), is_end=True)
        if s and e:
            start_m = s[0] * 12 + s[1]
            end_m = e[0] * 12 + e[1]
            if end_m >= start_m:
                intervals.append((start_m, end_m))

    if not intervals:
        return float(len(experience_detailed))

    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    total_months = sum(end - start for start, end in merged)
    return round(total_months / 12.0, 1)

def parse_skills_purified(skills_text: str, full_text: str):
    skills_text = normalize_text_dashes(skills_text)
    full_text = normalize_text_dashes(full_text)
    
    extracted = []
    
    junk_words = {
        "declaration", "interests", "languages", "english", "hindi", "odia", "reading", "painting",
        "cooking", "programming", "miscellaneous", "hobbies", "activities", "only", "player",
        "thoroughly", "concept", "made", "learned", "this", "using", "power", "features",
        "showcases", "working", "design", "visually", "format", "the", "true", "belief", "signature",
        "date", "place", "name", "cloud & tools", "databases", "data analysis", "visualization", "machine learning",
        "tools", "technologies", "technical skills", "core skills", "tech stack", "functional skills", "model", "server"
    }
    
    domain_terms = [
        "ATS", "Recruitment", "Talent Acquisition", "Sourcing", "Onboarding", "Screening",
        "Compliance", "Employee Relations", "Employer Branding", "HR Analytics", "SAP",
        "Workforce Planning", "Stakeholder Engagement", "Campus Hiring", "HRIS", "HRMS", "Payroll"
    ]
    for dt in domain_terms:
        if re.search(rf"\b{re.escape(dt)}\b", full_text, flags=re.I):
            extracted.append(dt)

    for line in skills_text.split('\n'):
        parts = re.split(r'[,|•●▪\t;]', line)
        for p in parts:
            item = p.strip().strip('()[]•●- \t\r\n')
            if not item:
                continue
            if len(item.split()) > 7 or len(item) > 55:
                continue
            if item.lower() in junk_words:
                continue
            if any(v in item.lower() for v in ["learned", "made this", "implemented", "utilizes", "taking"]):
                continue
            if len(item) >= 2:
                extracted.append(item)

    # Full text scan from normalizer alias map
    full_text_lower = full_text.lower()
    for alias in normalizer._alias_map.keys():
        if len(alias) >= 3 or alias in ["c", "r", "go", "ui", "ux", "ai", "ml", "db"]:
            if re.search(rf"\b{re.escape(alias)}\b", full_text_lower):
                if alias.lower() not in junk_words:
                    extracted.append(alias.title())

    clean_unique = []
    for s in extracted:
        norm = s.strip()
        if norm.lower() not in junk_words and norm not in clean_unique:
            clean_unique.append(norm)
            
    return normalizer.process_list(clean_unique)

class ResumeParser:
    @staticmethod
    def segment_sections(text: str):
        """Splits the document into logical sections using robust regex matching."""
        text = normalize_text_dashes(text)
        sections = {k: "" for k in HEADER_PATTERNS}
        sections["other"] = ""
        current_section = "other"
        
        lines = text.split('\n')
        for line in lines:
            clean_line = re.sub(r'^[^\w\s]+', '', line).strip()
            if not clean_line:
                continue
            
            is_header = False
            if len(clean_line.split()) < 6:
                is_all_caps = clean_line.isupper() and len(clean_line) > 3
                for sec_name, pattern in HEADER_PATTERNS.items():
                    if re.match(pattern, clean_line) or (is_all_caps and re.match(pattern, clean_line)):
                        current_section = sec_name
                        is_header = True
                        break
            
            if not is_header:
                sections[current_section] += line + "\n"
        
        return sections

    @staticmethod
    def parse(text: str):
        if not nlp:
            return {"error": "NLP model not loaded"}

        text = normalize_text_dashes(text)

        # 1. Section Segmentation
        sections = ResumeParser.segment_sections(text)
        
        # 2. NLP Processing
        doc = nlp(text)
        
        # 3. Contact Info (100% Offline NLP)
        emails = list(dict.fromkeys(re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)))
        
        raw_phone_candidates = re.findall(
            r"(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}(?:[\s.-]?\d{1,5})?",
            text
        )
        phones = []
        for cand in raw_phone_candidates:
            cand_clean = cand.strip(" \t\r\n-.,;:()")
            digits_only = re.sub(r"\D", "", cand_clean)
            if 10 <= len(digits_only) <= 15:
                if re.match(r"^(19|20)\d{2}(19|20)\d{2}$", digits_only):
                    continue
                if cand_clean not in phones:
                    phones.append(cand_clean)
        
        # 4. Name Extraction (100% Offline NLP)
        name = None
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        skip_keywords = [
            "resume", "cv", "curriculum", "curriculam", "vitale", "vitae", "profile", "summary",
            "contact", "email", "phone", "address", "objective", "career", "experience",
            "education", "skills", "projects", "certifications", "portfolio", "links",
            "page", "personal", "details", "information", "declaration"
        ]
        
        for line in lines[:15]:
            cleaned_line = re.sub(r"^[^\w\s]+", "", line).strip()
            lower_line = cleaned_line.lower()
            if not cleaned_line:
                continue
            if "@" in cleaned_line or any(kw in lower_line for kw in ["http", "www", "github", "linkedin"]):
                continue
            if len(re.findall(r'\d', cleaned_line)) > 2 or len(cleaned_line.split()) > 4 or len(cleaned_line.split()) < 2:
                continue
            if any(kw in lower_line for kw in skip_keywords):
                continue
            if any(re.match(r"(?i)^" + kw, lower_line) for kw in ["experience", "education", "skills", "about", "work", "technical"]):
                continue
            
            words = cleaned_line.split()
            if all(w[0].isupper() or w in ["de", "da", "von", "van", "al", "bin"] for w in words if w):
                if re.match(r"^[A-Za-z\s\.\-']+$", cleaned_line):
                    name = cleaned_line
                    break

        if not name:
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    clean_name = ent.text.strip().strip(" \t\r\n-.,;:")
                    words = clean_name.split()
                    if 2 <= len(words) <= 4 and not any(c.isdigit() for c in clean_name):
                        clean_lower = clean_name.lower()
                        if any(kw in clean_lower for kw in [
                            "limited", "company", "pvt", "ltd", "inc", "corp", "india", "university",
                            "college", "school", "institute", "hospital", "resume", "curriculum", "page"
                        ]):
                            continue
                        name = clean_name
                        break

        # 5. Extract Detailed Education & Experience
        education_detailed = parse_education_detailed(sections["education"])
        experience_detailed = parse_experience_detailed(sections["experience"])
        skills = parse_skills_purified(sections["skills"], text)

        extracted_locations = []
        extracted_companies = []
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                extracted_locations.append(ent.text.strip())
            elif ent.label_ == "ORG":
                extracted_companies.append(ent.text.strip())

        # Smart Company Filtering (using Normalizer)
        valid_companies = []
        for comp in extracted_companies:
            comp_id, comp_canon = normalizer.normalize_company(comp)
            if comp_id:
                valid_companies.append({"id": comp_id, "raw": comp, "canonical": comp_canon})

        # Smart Project Grouping
        projects_detailed = []
        current_proj = None
        for line in sections["projects"].split('\n'):
            line = line.strip()
            if not line: continue
            if line.isupper() and len(line) > 5 and len(line.split()) < 8:
                if current_proj: projects_detailed.append(current_proj)
                current_proj = {"title": line, "description": []}
            elif current_proj and line.startswith(('•', '-', '*', '●')):
                current_proj["description"].append(line)
            elif not current_proj and len(line) > 10:
                current_proj = {"title": "Project", "description": [line]}
        if current_proj: projects_detailed.append(current_proj)

        # Build detected roles from experience detailed
        detected_roles = [e["role"] for e in experience_detailed if e.get("role") and e["role"] != "Role"]
        institutes_and_degrees = []
        for e in education_detailed:
            if e.get("degree"): institutes_and_degrees.append(e["degree"])
            if e.get("school"): institutes_and_degrees.append(e["school"])

        # Extract header location & total experience years
        header_loc = extract_header_location(text)
        best_location_str = header_loc.get("raw") or ""
        exp_years = calculate_experience_years(text, experience_detailed)

        # Extract preferred locations if specified in text
        preferred_locations = []
        pref_match = re.search(r'(?i)preferred\s+locations?\s*:\s*([^\n|]+)', text)
        if pref_match:
            preferred_locations = [p.strip() for p in re.split(r'[/,;]', pref_match.group(1)) if p.strip()]

        initial_result = {
            "candidate_name": name or "Unknown",
            "location": best_location_str or "",
            "experience_years": exp_years,
            "preferred_locations": preferred_locations,
            "contact": {
                "emails": emails,
                "phones": phones
            },
            "experience": {
                "detected_roles": sorted(list(set(detected_roles)))[:10]
            },
            "experience_detailed": experience_detailed,
            "education": {
                "institutes_and_degrees": sorted(list(set(institutes_and_degrees)))[:10]
            },
            "education_detailed": education_detailed,
            "skills": skills,
            "skills_detailed": [{"name": s, "years_of_experience": (2 if len(experience_detailed) > 0 else 0)} for s in skills],
            "certifications": sorted(list(set([c.strip() for c in sections["certifications"].split('\n') if len(c.strip()) > 5]))),
            "projects": [f"{p['title']}: {' '.join(p['description'])}" for p in projects_detailed] if projects_detailed else [p.strip() for p in sections["projects"].split('\n') if len(p.strip()) > 10],
            "work_authorization": extract_work_authorization(text),
            "raw_text": text,
            "word_count": len(text.split())
        }

        refined_result = initial_result

        # ATS Normalization Post-Processing
        ats_data = {
            "location": {"id": None, "raw": None, "canonical": None, "city": None, "state": None, "country": None},
            "designations": [],
            "companies": valid_companies,
            "degrees": []
        }
        
        # Resolve location from header first, then spaCy entities
        if header_loc.get("city") or header_loc.get("raw"):
            cand_loc = header_loc.get("city") or header_loc.get("raw")
            loc_id, loc_canon = normalizer.normalize_location(cand_loc)
            ats_data["location"] = {
                "id": loc_id,
                "raw": header_loc.get("raw"),
                "canonical": loc_canon or header_loc.get("city"),
                "city": header_loc.get("city"),
                "state": header_loc.get("state"),
                "country": header_loc.get("country")
            }
        elif extracted_locations:
            for loc in extracted_locations:
                if len(loc) > 30: continue
                loc_id, loc_canon = normalizer.normalize_location(loc)
                if loc_id:
                    ats_data["location"] = {"id": loc_id, "raw": loc, "canonical": loc_canon, "city": loc, "state": "", "country": ""}
                    break
            if not ats_data["location"]["id"] and extracted_locations:
                ats_data["location"]["raw"] = extracted_locations[0]
            
        if not initial_result["location"] and ats_data["location"]["raw"]:
            initial_result["location"] = ats_data["location"]["raw"]

        roles_to_check = list(set(detected_roles))
        for role in roles_to_check:
            desig_id, desig_canon = normalizer.normalize_designation(role)
            ats_data["designations"].append({"id": desig_id, "raw": role, "canonical": desig_canon})
                
        degs_to_check = [e["degree"] for e in education_detailed if e.get("degree")]
        for deg in degs_to_check:
            deg_id, deg_canon = normalizer.normalize_degree(deg)
            ats_data["degrees"].append({"id": deg_id, "raw": deg, "canonical": deg_canon})
                
        refined_result["ats_normalized"] = ats_data
        
        return refined_result


class JobDescriptionParser:
    @staticmethod
    def parse(text: str):
        # 1. Local NLP & Section Segmentation (Ceipal Style)
        lines = text.split('\n')
        
        required_lines = []
        preferred_lines = []
        current_section = "required" # default is required/core
        
        # Ceipal Header keywords
        required_headers = ["required", "must have", "minimum", "essential", "responsibilities", "duties", "key skills", "requirements"]
        preferred_headers = ["preferred", "nice to have", "desired", "plus", "optional", "pluses", "secondary", "bonus"]
        
        for line in lines:
            clean = line.strip().lower()
            if not clean:
                continue
            
            # Clean markdown symbols (#, *, -, etc.) from line for header checking
            clean_header = re.sub(r'^[#\*\-\s]+', '', clean).strip()
            
            # Check if line is a short section header
            if len(clean_header.split()) < 6:
                is_req = any(h in clean_header for h in required_headers)
                is_pref = any(h in clean_header for h in preferred_headers)
                
                if is_pref:
                    current_section = "preferred"
                    continue
                elif is_req:
                    current_section = "required"
                    continue
            
            if current_section == "required":
                required_lines.append(line)
            else:
                preferred_lines.append(line)
                
        required_text = "\n".join(required_lines).lower()
        preferred_text = "\n".join(preferred_lines).lower()
        
        # Extract skills for each zone
        from app.normalizer import normalizer
        primary_skills_set = set()
        secondary_skills_set = set()
        
        # a. Dictionary Scan for Required Zone (Primary Skills)
        for alias, canon in normalizer._alias_map.items():
            if f" {alias} " in f" {required_text} " or f"\n{alias}\n" in f"\n{required_text}\n" or required_text.startswith(alias) or required_text.endswith(alias):
                primary_skills_set.add(canon)
                
        # b. Dictionary Scan for Preferred Zone (Secondary Skills)
        for alias, canon in normalizer._alias_map.items():
            if f" {alias} " in f" {preferred_text} " or f"\n{alias}\n" in f"\n{preferred_text}\n" or preferred_text.startswith(alias) or preferred_text.endswith(alias):
                if canon not in primary_skills_set:
                    secondary_skills_set.add(canon)
                    
        # c. spaCy Match (distribute based on positional split)
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "SKILL":
                    canon = normalizer.normalize(ent.text) or ent.text.title()
                    if ent.text.lower() in required_text:
                        primary_skills_set.add(canon)
                        secondary_skills_set.discard(canon)
                    elif ent.text.lower() in preferred_text and canon not in primary_skills_set:
                        secondary_skills_set.add(canon)

        local_primary = sorted(list(primary_skills_set))
        local_secondary = sorted(list(secondary_skills_set))
        
        # Try to guess job title locally
        local_title = "Unknown"
        
        # First pass: Look for explicit labels anywhere
        for line in lines:
            match = re.match(r"(?i)^(?:role|job title|position)\s*:\s*(.+)", line.strip())
            if match and match.group(1).strip():
                local_title = match.group(1).strip()
                break
                
        # Second pass: Fallback to top lines
        if local_title == "Unknown":
            for line in lines[:4]:
                clean_line = line.strip()
                if len(clean_line.split()) < 8 and any(kw in clean_line.lower() for kw in ["engineer", "developer", "manager", "lead", "architect", "designer", "consultant"]):
                    local_title = clean_line
                    break
                
        # Try to detect work auth locally
        local_work_auth = extract_work_authorization(text)
        
        result = {
            "jobTitle": local_title,
            "primarySkills": local_primary,
            "secondarySkills": local_secondary,
            "workAuthorization": local_work_auth,
            "roles": []
        }
        
        # 2. 100% Offline NLP Mode: Balance primary & secondary skills locally
        result["primarySkills"] = local_primary
        result["secondarySkills"] = [s for s in local_secondary if s not in result["primarySkills"]]
        return result

