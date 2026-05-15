import re

def test_split(keywords):
    clean_keywords = re.sub(r'(?i)\b(and|or|with)\b', ',', keywords)
    clean_keywords = clean_keywords.replace('&', ',').replace('+', ',').replace('/', ',')
    
    if ',' not in clean_keywords and '|' not in clean_keywords:
        kw_list = [k.strip().lower() for k in clean_keywords.split() if k.strip()]
    else:
        kw_list = [k.strip().lower() for k in re.split(r'[,|]+', clean_keywords) if k.strip()]
    return kw_list

keywords = "python & sql & node.js & Javascript"
print(f"Keywords: {keywords}")
print(f"Split result: {test_split(keywords)}")

search_blob = "{'skills': ['Javascript', 'Node.Js', 'Python', 'Sql']}".lower()
print(f"Search Blob: {search_blob}")

kw_list = test_split(keywords)
match = all(kw in search_blob for kw in kw_list)
print(f"Match: {match}")
