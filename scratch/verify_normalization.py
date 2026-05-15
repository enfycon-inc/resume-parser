from app.normalizer import normalizer

def test_normalization():
    input_skills = ["ReactJS", "Aws", "NodeJS", "Js", "K8s", "Docker", "python"]
    print(f"Input: {input_skills}")
    
    output = normalizer.process_list(input_skills)
    print(f"Output (Canonical): {output}")
    
    # Expected: ['AWS', 'Docker', 'JavaScript', 'Kubernetes', 'Node.js', 'Python', 'React']

if __name__ == "__main__":
    test_normalization()
