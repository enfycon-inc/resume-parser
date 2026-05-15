from database import SessionLocal, SkillMaster, SkillAlias
import logging

logger = logging.getLogger(__name__)

class SkillNormalizer:
    _instance = None
    _alias_map = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SkillNormalizer, cls).__new__(cls)
            cls._instance._load_dictionary()
        return cls._instance

    def _load_dictionary(self):
        """Loads all aliases into a RAM-based hashmap from the database."""
        db = SessionLocal()
        try:
            # Query all aliases with their canonical names
            query = db.query(SkillAlias.alias_name, SkillMaster.canonical_name)\
                      .join(SkillMaster, SkillAlias.skill_id == SkillMaster.id).all()
            
            self._alias_map = {row.alias_name: row.canonical_name for row in query}
            logger.info(f"Skill Normalizer loaded {len(self._alias_map)} aliases into RAM.")
        except Exception as e:
            logger.error(f"Failed to load skill dictionary: {e}")
        finally:
            db.close()

    def normalize(self, raw_skill: str):
        """Maps a raw skill string to its canonical version."""
        return self._alias_map.get(raw_skill.lower().strip())

    def process_list(self, raw_skills: list):
        """Normalizes a list of skills and returns unique canonical names."""
        canonical_skills = set()
        for s in raw_skills:
            normalized = self.normalize(s)
            if normalized:
                canonical_skills.add(normalized)
            else:
                # Fallback: if not in dictionary, title case it
                if len(s) > 1:
                    canonical_skills.add(s.title())
        
        return sorted(list(canonical_skills))

# Singleton instance
normalizer = SkillNormalizer()
