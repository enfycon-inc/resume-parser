from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def generate_embedding(self, text: str):
        """Generates a 384-dimensional vector for the given text."""
        if not self.model:
            return None
        
        # We truncate long resumes to avoid context window issues
        truncated_text = text[:2000] 
        embedding = self.model.encode(truncated_text)
        return embedding.tolist()

    def generate_resume_vector(self, parsed_data: dict):
        """
        Creates a weighted vector based on Skills and Experience.
        This is what we store for Semantic Search.
        """
        # We combine skills and roles into a single string for better semantic context
        skills_str = ", ".join(parsed_data.get("skills", []))
        roles_str = ", ".join(parsed_data.get("experience", {}).get("detected_roles", []))
        
        combined_text = f"Skills: {skills_str}. Experience: {roles_str}."
        return self.generate_embedding(combined_text)
