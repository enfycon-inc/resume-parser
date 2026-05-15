from database import SessionLocal, SkillCategory, SkillMaster, SkillAlias, init_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SKILL_DATA = {
    "Programming Language": {
        "Python": ["python", "py", "python3", "python2"],
        "Java": ["java", "j2ee", "core java"],
        "JavaScript": ["js", "javascript", "es6", "es5"],
        "TypeScript": ["ts", "typescript"],
        "C++": ["cpp", "cplusplus", "c++", "c/c++"],
        "SQL": ["sql", "sql query", "t-sql"],
        "C#": ["csharp", "c#", ".net c#"],
        "PHP": ["php", "php7", "php8"],
        "Go": ["golang", "go"],
        "Ruby": ["ruby", "rb", "ror"],
        "Swift": ["swift", "swiftui"],
        "Kotlin": ["kotlin"],
        "Rust": ["rust"],
        "Scala": ["scala"]
    },
    "Frontend": {
        "React": ["react", "reactjs", "react.js", "react js"],
        "Angular": ["angular", "angularjs", "angular.js"],
        "Vue": ["vue", "vuejs", "vue.js"],
        "Next.js": ["nextjs", "next.js"],
        "Tailwind CSS": ["tailwind", "tailwindcss"],
        "Bootstrap": ["bootstrap", "bootstrap5"],
        "Sass": ["scss", "sass"],
        "Redux": ["redux", "redux-toolkit"],
        "Flutter": ["flutter", "dart"]
    },
    "Backend": {
        "Node.js": ["node", "nodejs", "node.js", "node js"],
        "Django": ["django"],
        "Flask": ["flask"],
        "FastAPI": ["fastapi"],
        "Spring Boot": ["spring", "springboot", "spring boot"],
        "Express.js": ["express", "expressjs"],
        "Laravel": ["laravel"],
        "Rails": ["rails", "ruby on rails"],
        "GraphQL": ["graphql"]
    },
    "Cloud": {
        "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda"],
        "Azure": ["azure", "microsoft azure"],
        "Google Cloud": ["gcp", "google cloud platform"],
        "Heroku": ["heroku"]
    },
    "Database": {
        "PostgreSQL": ["postgres", "postgresql", "psql"],
        "MySQL": ["mysql", "mariadb"],
        "MongoDB": ["mongo", "mongodb"],
        "Redis": ["redis"],
        "Elasticsearch": ["elk", "elasticsearch"],
        "Oracle": ["oracle db", "oracle sql"],
        "SQL Server": ["mssql", "sql server"]
    },
    "DevOps": {
        "Docker": ["docker"],
        "Kubernetes": ["k8s", "kubernetes", "kube"],
        "Terraform": ["terraform", "tf", "SEACOM"],
        "Jenkins": ["jenkins"],
        "Ansible": ["ansible"],
        "Git": ["git", "github", "gitlab", "bitbucket"],
        "Linux": ["linux", "ubuntu", "debian", "centos", "redhat"]
    },
    "AI/ML": {
        "TensorFlow": ["tensorflow", "tf"],
        "PyTorch": ["pytorch"],
        "Scikit-learn": ["sklearn", "scikit-learn"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "LangChain": ["langchain"],
        "OpenAI": ["gpt", "openai", "llm"],
        "Hugging Face": ["huggingface", "transformers"]
    },
    "Tools & Others": {
        "Latex": ["latex", "tex"],
        "Tableau": ["tableau"],
        "Microsoft Office": ["ms office", "microsoft office", "excel", "powerpoint", "word"],
        "Firebase": ["firebase", "google firebase"],
        "HTML/CSS": ["html/css", "html css"]
    }
}

def seed_skills():
    init_db()
    db = SessionLocal()
    try:
        for cat_name, skills in SKILL_DATA.items():
            # Create Category
            cat = db.query(SkillCategory).filter(SkillCategory.name == cat_name).first()
            if not cat:
                cat = SkillCategory(name=cat_name)
                db.add(cat)
                db.flush()
            
            for canonical, aliases in skills.items():
                # Create Skill Master
                skill = db.query(SkillMaster).filter(SkillMaster.canonical_name == canonical).first()
                if not skill:
                    skill = SkillMaster(canonical_name=canonical, category_id=cat.id)
                    db.add(skill)
                    db.flush()
                
                # Add Aliases
                # Include canonical name as an alias too
                if canonical.lower() not in aliases:
                    aliases.append(canonical.lower())
                
                for alias in aliases:
                    exists = db.query(SkillAlias).filter(SkillAlias.alias_name == alias.lower()).first()
                    if not exists:
                        db.add(SkillAlias(skill_id=skill.id, alias_name=alias.lower()))
        
        db.commit()
        logger.info("Skill Dictionary seeded successfully with 200+ aliases!")
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_skills()
