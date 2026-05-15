from sqlalchemy import create_engine, text
import logging

DATABASE_URL = "postgresql://postgres:EWhbqnM6IWe5IJaV@db.zqpxnnsbbqdlhememsyj.supabase.co:5432/postgres"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def drop_unique_constraint():
    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as conn:
            logger.info("Dropping unique constraint on file_hash...")
            # In Postgres, SQLAlchemy usually names it like this: table_column_key
            conn.execute(text("ALTER TABLE resumes DROP CONSTRAINT IF EXISTS resumes_file_hash_key"))
            conn.commit()
            logger.info("Constraint dropped successfully.")
    except Exception as e:
        logger.error(f"Failed to drop constraint: {e}")

if __name__ == "__main__":
    drop_unique_constraint()
