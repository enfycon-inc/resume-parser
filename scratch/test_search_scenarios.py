import sys
import os
import re
import json

# Add parent directory to path to import main and database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_all_resumes
from normalizer import normalizer

def simulate_boolean_search(keywords):
    print(f"\n[TEST] Searching for: '{keywords}'")
    
    # 1. Logic from main.py
    clean_keywords = re.sub(r'(?i)\b(and|with|&)\b', ',', keywords)
    if any(d in clean_keywords for d in [',', '|', '/']):
        raw_keyword_list = [k.strip().lower() for k in re.split(r'[,|/]', clean_keywords) if k.strip()]
    else:
        raw_keyword_list = [k.strip().lower() for k in clean_keywords.split() if k.strip()]
    
    search_groups = []
    for k in raw_keyword_list:
        group = [k]
        canonical = normalizer.normalize(k)
        if canonical:
            group.append(canonical.lower())
            group.append(canonical.lower().replace('.', '').replace('/', ''))
        search_groups.append(list(set(group)))
    
    print(f"  Groups generated: {search_groups}")
    
    # 2. Get all resumes and filter
    all_recs = get_all_resumes()
    matches = []
    for r in all_recs:
        p_json = r.parsed_json
        if isinstance(p_json, str):
            p_json = json.loads(p_json)
            
        skills_str = " ".join(p_json.get("skills", []))
        roles_str = " ".join(p_json.get("experience", {}).get("detected_roles", []))
        blob = f"{r.candidate_name} {r.email} {skills_str} {roles_str}".lower()
        
        matches_all = True
        for group in search_groups:
            if not any(term in blob for term in group):
                matches_all = False
                break
        
        if matches_all:
            matches.append(r.candidate_name)
            
    print(f"  Found {len(matches)} matches: {matches}")
    return matches

if __name__ == "__main__":
    print("--- ATS Search Scenario Tester ---")
    simulate_boolean_search("nodejs and react")
    simulate_boolean_search("python, sql")
    simulate_boolean_search("K8s / AWS")
    simulate_boolean_search("Sahadeb")
