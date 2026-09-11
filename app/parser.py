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

class ResumeParser:
    @staticmethod
    def segment_sections(text: str):
        """Splits the document into logical sections using robust regex matching."""
        sections = {
            "experience": "",
            "education": "",
            "skills": "",
            "projects": "",
            "certifications": "",
            "other": ""
        }
        
        # Comprehensive header patterns (Refined to be standalone lines or end with colon)
        header_patterns = {
            "experience": r"(?i)^\s*(experience|employment|work history|professional experience|background|roles & responsibilities|professional background|experience\d+|work\s+experience)\s*[:]?\s*$",
            "education": r"(?i)^\s*(education|academic|scholastic|qualifications|academic\s+history)\s*[:]?\s*$",
            "skills": r"(?i)^\s*(skills|technologies|proficiencies|expertise|technical skills|tools|key\s+skills|core\s+competencies|competencies|programming|miscellaneous)\s*[:]?\s*$",
            "certifications": r"(?i)^\s*(certifications|certificates|training|courses|awards|certification)\s*[:]?\s*$",
            "projects": r"(?i)^\s*(projects|assignments|key projects|academic projects|recent projects|portfolio)\s*[:]?\s*$"
        }
        
        current_section = "other"
        lines = text.split('\n')
        
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue
            
            # Check if line is a header (usually short, standalone or ends with colon)
            is_header = False
            if len(clean_line.split()) < 6:
                # Check for regex match
                # Also consider all-caps lines as headers if they contain section keywords
                is_all_caps = clean_line.isupper() and len(clean_line) > 3
                for sec_name, pattern in header_patterns.items():
                    if re.match(pattern, clean_line) or (is_all_caps and re.match(pattern, clean_line)):
                        current_section = sec_name
                        is_header = True
                        break
            
            # Specialized check for common "Role:" headers in experience
            if not is_header and "experience" in current_section:
                if re.match(r"(?i)^(roles?|responsibilities|job|position|duties):", clean_line):
                    # Keep it in experience
                    pass

            if not is_header:
                sections[current_section] += clean_line + "\n"
        
        return sections

    @staticmethod
    def parse(text: str):
        if not nlp:
            return {"error": "NLP model not loaded"}

        # 1. Section Segmentation
        sections = ResumeParser.segment_sections(text)
        
        # 2. NLP Processing
        doc = nlp(text)
        
        # 3. Contact Info (100% Offline NLP)
        emails = list(dict.fromkeys(re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)))
        
        # Phone numbers: support international codes (+91, +1), spaces, dashes, brackets
        raw_phone_candidates = re.findall(
            r"(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}(?:[\s.-]?\d{1,5})?",
            text
        )
        phones = []
        for cand in raw_phone_candidates:
            cand_clean = cand.strip(" \t\r\n-.,;:()")
            digits_only = re.sub(r"\D", "", cand_clean)
            if 10 <= len(digits_only) <= 15:
                # Discard years/date ranges like 2014-2018 or 20202024
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
            # Remove any leading icons/symbols (e.g. , , bullet points)
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


        # 5. Extract Skills (Section + Entities + Dictionary Scan)
        extracted_skills = []
        # a. From Skills Section
        skill_lines = sections["skills"].split('\n')
        for line in skill_lines:
            parts = re.split(r'[,|•\t;]', line)
            for p in parts:
                if len(p.strip()) > 1:
                    extracted_skills.append(p.strip())

        # b. From Entities
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                extracted_skills.append(ent.text)
            
        # c. Full-Text Dictionary Scan (Critical for "SEACOM" test without AI)
        # This scans the whole text for anything in our alias map
        from app.normalizer import normalizer
        full_text_lower = text.lower()
        for alias in normalizer._alias_map.keys():
            # Look for whole word matches to avoid partial matching (e.g. "py" in "happy")
            if re.search(rf"\b{re.escape(alias)}\b", full_text_lower):
                extracted_skills.append(alias)

        extracted_degrees = []
        extracted_institutes = []
        extracted_roles = []
        extracted_locations = []
        extracted_companies = []
        
        for ent in doc.ents:
            if ent.label_ == "DEGREE":
                extracted_degrees.append(ent.text.strip())
            elif ent.label_ in ["GPE", "LOC"]:
                extracted_locations.append(ent.text.strip())
            elif ent.label_ == "ORG":
                extracted_companies.append(ent.text.strip())

        # 6. Refine Skills
        if sections["skills"]:
            skill_lines = re.split(r'[\n•\-\*]', sections["skills"])
            for sl in skill_lines:
                clean_line = sl.strip()
                if not clean_line or len(clean_line) > 150: continue
                
                # Split by comma ONLY if it doesn't look like a complex sentence (has parentheses or fewer than 2 commas)
                if ',' in clean_line and '(' not in clean_line and clean_line.count(',') < 5:
                    parts = clean_line.split(',')
                else:
                    parts = [clean_line]
                
                for p in parts:
                    s = p.strip().strip('()[]•●- \t\r\n')
                    if 2 <= len(s) <= 45 and len(s.split()) < 7:
                        # Stricter filtering for fragments
                        if not any(kw in s.lower() for kw in ["via", "focused", "reports", "signature", "date:", "wide", "enabled", "coordinated"]):
                            extracted_skills.append(s)

        # 7. Refine Education
        if sections["education"]:
            edu_doc = nlp(sections["education"])
            for ent in edu_doc.ents:
                if ent.label_ == "ORG":
                    # Validate that it looks like an educational institute
                    text_lower = ent.text.lower()
                    if any(kw in text_lower for kw in ["university", "college", "school", "institute", "vidya", "mandir", "academy", "learning", "education"]):
                        extracted_institutes.append(ent.text.strip())
            
            for line in sections["education"].split('\n'):
                l_lower = line.lower()
                # Ensure it's a degree line and not just a skill mention
                if any(kw in l_lower for kw in ["graduation", "bachelor", "master", "degree", "post grad", "mca", "mba", "b.tech", "m.tech", "diploma"]):
                    # Basic filter to avoid skills leaking in if they happen to have those keywords
                    if len(line.split()) < 10:
                        extracted_degrees.append(line.strip())

        # 8. Refine Experience & Extract Temporal Dates
        experience_detailed = []
        for sec_name in ["experience"]:
            sec_text = sections[sec_name]
            if not sec_text: continue
            
            exp_lines = [l.strip() for l in sec_text.split('\n') if len(l.strip()) > 5]
            current_role = None
            current_company = None
            
            for line in exp_lines:
                # 8a. Detect Roles
                role_match = re.search(r"\b(manager|engineer|consultant|analyst|developer|lead|assistant|executive|specialist|officer|coordinator|head|director|intern|trainee|recruiter|generalist)\b", line, re.I)
                if role_match and len(line.split()) < 10 and not line.startswith(('●', '•', '-', '*')):
                    extracted_roles.append(line)
                    current_role = line
                
                # 8b. Detect Dates
                date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{2,4}\s*[-–to]+\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present|Current)[a-z]*\s*\d{0,4}', line, re.I)
                if not date_match:
                    # Try YYYY - YYYY
                    date_match = re.search(r'\b(19|20)\d{2}\s*[-–to]+\s*(19|20)\d{2}|Present\b', line, re.I)

                if current_role and date_match:
                    date_str = date_match.group(0)
                    start_date = date_str.split('-')[0].strip() if '-' in date_str else date_str
                    end_date = date_str.split('-')[1].strip() if '-' in date_str else "Present"
                    experience_detailed.append({
                        "role": current_role,
                        "company": "Unknown", # Needs DB normalizer to accurately guess
                        "start_date": start_date,
                        "end_date": end_date,
                        "is_current": "present" in end_date.lower() or "current" in end_date.lower()
                    })
                    current_role = None # reset to avoid duplicate attachment

        # 9. Smart Company Filtering (using Normalizer)
        valid_companies = []
        for comp in extracted_companies:
            comp_id, comp_canon = normalizer.normalize_company(comp)
            if comp_id: # Only trust it if it actually exists in our master dictionary!
                valid_companies.append({"id": comp_id, "raw": comp, "canonical": comp_canon})

        # 10. Smart Project Grouping
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

        # Final Structured Result
        initial_result = {
            "candidate_name": name or "Unknown",
            "contact": {
                "emails": emails,
                "phones": phones
            },
            "experience": {
                "detected_roles": sorted(list(set([r.strip() for r in extracted_roles if len(r.strip()) > 3])))[:10]
            },
            "experience_detailed": experience_detailed,
            "education": {
                "institutes_and_degrees": sorted(list(set([e.strip() for e in (extracted_degrees + extracted_institutes) if len(e.strip()) > 3])))[:10]
            },
            "skills": sorted(list(set([s.strip().title() for s in extracted_skills if len(s.strip()) > 1]))),
            "certifications": sorted(list(set([c.strip() for c in sections["certifications"].split('\n') if len(c.strip()) > 5]))),
            "projects": [f"{p['title']}: {' '.join(p['description'])}" for p in projects_detailed] if projects_detailed else [p.strip() for p in sections["projects"].split('\n') if len(p.strip()) > 10],
            "work_authorization": extract_work_authorization(text),
            "word_count": len(text.split())
        }

        # Local Junk Filter (Post-AI Cleanup)
        junk_skills = {"Only", "Player", "Thoroughly", "Concept", "Made", "Learned", "This", "Using", "Power", "Features", "Showcases", "Working", "Design", "Visually", "Format", "Searchforfoodrecipes", "The", "Thoroughly", "Learned Frontend Web Development Thoroughly"}
        if "skills" in initial_result:
            clean_list = []
            for s in initial_result["skills"]:
                if s in junk_skills: continue
                if len(s) > 40 or len(s.split()) > 4: continue
                clean_list.append(s)
            
            normalized_skills = normalizer.process_list(clean_list)
            initial_result["skills"] = normalized_skills
            # Map years of experience to skills using a simple heuristic since AI isn't here
            years = 2 if len(experience_detailed) > 0 else 0
            initial_result["skills_detailed"] = [{"name": s, "years_of_experience": years} for s in normalized_skills]

        # 100% Offline NLP Mode: No LLM/AI refinement
        refined_result = initial_result

        
        # ATS Normalization Post-Processing
        ats_data = {
            "location": {"id": None, "raw": None},
            "designations": [],
            "companies": valid_companies,
            "degrees": []
        }
        
        for loc in extracted_locations:
            if len(loc) > 30: continue
            loc_id, loc_canon = normalizer.normalize_location(loc)
            if loc_id:
                ats_data["location"] = {"id": loc_id, "raw": loc, "canonical": loc_canon}
                break
        if not ats_data["location"]["id"] and extracted_locations:
            ats_data["location"]["raw"] = extracted_locations[0]
            
        roles_to_check = list(set(extracted_roles))
        if "experience" in refined_result and isinstance(refined_result["experience"], list):
            roles_to_check.extend([str(r) for r in refined_result["experience"] if isinstance(r, str)])
        roles_to_check = list(set(roles_to_check))
        
        for role in roles_to_check:
            desig_id, desig_canon = normalizer.normalize_designation(role)
            ats_data["designations"].append({"id": desig_id, "raw": role, "canonical": desig_canon})
                
        degs_to_check = list(set(extracted_degrees))
        if "education" in refined_result and isinstance(refined_result["education"], list):
             degs_to_check.extend([str(d) for d in refined_result["education"] if isinstance(d, str)])
        degs_to_check = list(set(degs_to_check))
        
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
                    # Check if entity index is in first half of the document text
                    if ent.start_char < len(text) // 2:
                        primary_skills_set.add(canon)
                    else:
                        if canon not in primary_skills_set:
                            secondary_skills_set.add(canon)
                            
        local_primary = sorted(list(primary_skills_set))[:6]
        local_secondary = sorted(list(secondary_skills_set))[:6]
        
        # Try to guess job title locally
        local_title = "Unknown"
        
        # First pass: Look for explicit labels anywhere
        import re
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
        result["primarySkills"] = local_primary[:8]
        result["secondarySkills"] = [s for s in local_secondary if s not in result["primarySkills"]][:8]
        return result

