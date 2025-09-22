import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import re
import pickle
import os
import hashlib
import warnings
from typing import List, Dict, Tuple, Optional, Union
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor
import joblib
from pathlib import Path
import sqlite3
from datetime import datetime
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------
# Configuration and Constants
# ----------------------------
class Config:
    CACHE_DIR = "embeddings_cache"
    MODEL_NAME = 'all-MiniLM-L6-v2'
    DEFAULT_TOP_N = 25
    MIN_SCORE_THRESHOLD = 0.1
    MAX_WORKERS = 4
    USER_FEEDBACK_DB = "user_feedback.db"
    
    # Skill mappings for normalization
    SKILL_MAPPING = {
        'javascript': ['js', 'javascript', 'java script'],
        'python': ['python', 'py'],
        'java': ['java'],
        'c++': ['c++', 'cpp', 'c plus plus'],
        'c': ['c', 'c language'],
        'html': ['html', 'html5'],
        'css': ['css', 'css3', 'cascading style sheets'],
        'react': ['react', 'reactjs', 'react.js'],
        'node': ['node', 'nodejs', 'node.js'],
        'sql': ['sql', 'mysql', 'postgresql', 'sqlite'],
        'mongodb': ['mongodb', 'mongo'],
        'git': ['git', 'github', 'version control'],
        'machine learning': ['ml', 'machine learning', 'ai'],
        'data science': ['data science', 'data analysis', 'analytics'],
        'android': ['android', 'android development'],
        'ios': ['ios', 'ios development', 'swift'],
        'flutter': ['flutter', 'dart'],
        'docker': ['docker', 'containerization'],
        'aws': ['aws', 'amazon web services'],
        'azure': ['azure', 'microsoft azure'],
        'gcp': ['gcp', 'google cloud', 'google cloud platform']
    }
    
    # Scoring weights (adjusted to include collaborative filtering)
    WEIGHTS = {
        'skills_exact': 0.35,        # Reduced to make room for CF
        'skills_semantic': 0.15,     # Reduced to make room for CF
        'location': 0.25,            # Reduced to make room for CF
        'collaborative': 0.20,       # New collaborative filtering weight
        'popularity': 0.05,          # Basic popularity score
        'gender': 0.0,
        'payment': 0.0
    }

# ----------------------------
# Database Manager for User Feedback
# ----------------------------
class FeedbackManager:
    """Manages user feedback storage and retrieval."""
    
    def __init__(self, db_path: str = Config.USER_FEEDBACK_DB):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the feedback database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                user_profile TEXT,  -- JSON string of user skills/preferences
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feedback table with unique constraint
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                internship_id TEXT,
                feedback_type TEXT,  -- 'upvote', 'downvote', 'apply', 'skip'
                rating INTEGER,      -- 1-5 scale (optional)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                UNIQUE(user_id, internship_id, feedback_type)
            )
        ''')
        
        # User similarity cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_similarity (
                user1_id TEXT,
                user2_id TEXT,
                similarity_score REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user1_id, user2_id)
            )
        ''')
        
        # Internship popularity cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS internship_stats (
                internship_id TEXT PRIMARY KEY,
                total_upvotes INTEGER DEFAULT 0,
                total_downvotes INTEGER DEFAULT 0,
                total_applications INTEGER DEFAULT 0,
                avg_rating REAL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_user(self, user_id: str, user_profile: Dict):
        """Add or update user profile."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO users (user_id, user_profile)
            VALUES (?, ?)
        ''', (user_id, json.dumps(user_profile)))
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, user_id: str, internship_id: str, feedback_type: str, rating: Optional[int] = None):
        """Add user feedback for an internship with duplicate prevention."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Use INSERT OR REPLACE to handle duplicates
            cursor.execute('''
                INSERT OR REPLACE INTO feedback (user_id, internship_id, feedback_type, rating)
                VALUES (?, ?, ?, ?)
            ''', (user_id, internship_id, feedback_type, rating))
            
            # Update internship stats
            self._update_internship_stats(cursor, internship_id)
            
            conn.commit()
            logging.info(f"Added/updated feedback: {user_id} -> {internship_id} ({feedback_type})")
            
        except sqlite3.IntegrityError as e:
            logging.warning(f"Duplicate feedback ignored: {user_id} -> {internship_id} ({feedback_type})")
            conn.rollback()
        finally:
            conn.close()
    
    def _update_internship_stats(self, cursor, internship_id: str):
        """Update internship popularity statistics."""
        # Get current stats
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN feedback_type = 'upvote' THEN 1 ELSE 0 END) as upvotes,
                SUM(CASE WHEN feedback_type = 'downvote' THEN 1 ELSE 0 END) as downvotes,
                SUM(CASE WHEN feedback_type = 'apply' THEN 1 ELSE 0 END) as applications,
                AVG(CASE WHEN rating IS NOT NULL THEN rating END) as avg_rating
            FROM feedback 
            WHERE internship_id = ?
        ''', (internship_id,))
        
        stats = cursor.fetchone()
        upvotes, downvotes, applications, avg_rating = stats
        
        cursor.execute('''
            INSERT OR REPLACE INTO internship_stats 
            (internship_id, total_upvotes, total_downvotes, total_applications, avg_rating)
            VALUES (?, ?, ?, ?, ?)
        ''', (internship_id, upvotes or 0, downvotes or 0, applications or 0, avg_rating or 0))
    
    def get_user_feedback_matrix(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Get user-internship feedback matrix."""
        conn = sqlite3.connect(self.db_path)
        
        # Get feedback data with ratings
        query = '''
            SELECT user_id, internship_id, feedback_type, rating
            FROM feedback
            ORDER BY created_at DESC
        '''
        
        feedback_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if feedback_df.empty:
            return pd.DataFrame(), [], []
        
        # Convert feedback to numerical scores
        def feedback_to_score(row):
            if row['feedback_type'] == 'upvote':
                return row['rating'] if row['rating'] else 4
            elif row['feedback_type'] == 'downvote':
                return row['rating'] if row['rating'] else 1
            elif row['feedback_type'] == 'apply':
                return 5
            elif row['feedback_type'] == 'skip':
                return 2
            else:
                return row['rating'] if row['rating'] else 3
        
        feedback_df['score'] = feedback_df.apply(feedback_to_score, axis=1)
        
        # Create user-item matrix
        user_item_matrix = feedback_df.pivot_table(
            index='user_id', 
            columns='internship_id', 
            values='score', 
            aggfunc='mean'
        ).fillna(0)
        
        return user_item_matrix, list(user_item_matrix.index), list(user_item_matrix.columns)
    
    def get_internship_popularity_scores(self) -> Dict[str, float]:
        """Get popularity scores for internships."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT internship_id, total_upvotes, total_downvotes, total_applications, avg_rating
            FROM internship_stats
        ''')
        
        scores = {}
        for row in cursor.fetchall():
            internship_id, upvotes, downvotes, applications, avg_rating = row
            
            # Calculate popularity score combining multiple factors
            total_interactions = upvotes + downvotes + applications
            if total_interactions > 0:
                upvote_ratio = upvotes / total_interactions
                application_ratio = applications / total_interactions
                rating_score = (avg_rating / 5.0) if avg_rating else 0.5
                
                # Weighted popularity score
                popularity = (0.4 * upvote_ratio + 
                             0.4 * application_ratio + 
                             0.2 * rating_score)
                scores[internship_id] = popularity
            else:
                scores[internship_id] = 0.5  # Neutral score for new items
        
        conn.close()
        return scores

class CollaborativeFilter:
    """Collaborative Filtering implementation."""
    
    def __init__(self, feedback_manager: FeedbackManager):
        self.feedback_manager = feedback_manager
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.nmf_model = None
        self.user_factors = None
        self.item_factors = None
        self.user_list = []
        self.item_list = []
        self.is_trained = False
    
    def train_model(self, min_interactions: int = 5):
        """Train collaborative filtering model."""
        logging.info("Training collaborative filtering model...")
        
        user_item_matrix, users, items = self.feedback_manager.get_user_feedback_matrix()
        
        if user_item_matrix.empty or len(users) < 2:
            logging.warning("Insufficient data for collaborative filtering")
            self.is_trained = False
            return
        
        self.user_list = users
        self.item_list = items
        
        # Filter users and items with minimum interactions
        user_counts = (user_item_matrix > 0).sum(axis=1)
        item_counts = (user_item_matrix > 0).sum(axis=0)
        
        active_users = user_counts[user_counts >= min_interactions].index
        popular_items = item_counts[item_counts >= min_interactions].index
        
        if len(active_users) < 2 or len(popular_items) < 2:
            logging.warning("Insufficient active users or popular items")
            self.is_trained = False
            return
        
        # Filter matrix
        filtered_matrix = user_item_matrix.loc[active_users, popular_items]
        
        # Train NMF model for matrix factorization
        n_components = min(10, len(active_users) - 1, len(popular_items) - 1)
        self.nmf_model = NMF(n_components=n_components, random_state=42, max_iter=200)
        
        # Fit the model
        self.user_factors = self.nmf_model.fit_transform(filtered_matrix.values)
        self.item_factors = self.nmf_model.components_.T
        
        # Compute user similarity matrix
        self.user_similarity_matrix = cosine_similarity(self.user_factors)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity_matrix, 
            index=active_users, 
            columns=active_users
        )
        
        # Compute item similarity matrix
        self.item_similarity_matrix = cosine_similarity(self.item_factors)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity_matrix, 
            index=popular_items, 
            columns=popular_items
        )
        
        self.is_trained = True
        logging.info(f"Collaborative filtering model trained with {len(active_users)} users and {len(popular_items)} items")
    
    def get_user_based_recommendations(self, user_id: str, internship_ids: List[str], top_k: int = 5) -> Dict[str, float]:
        """Get user-based collaborative filtering scores."""
        if not self.is_trained or user_id not in self.user_similarity_df.index:
            return {iid: 0.5 for iid in internship_ids}
        
        # Find similar users
        user_similarities = self.user_similarity_df.loc[user_id].sort_values(ascending=False)[1:top_k+1]
        
        if user_similarities.empty:
            return {iid: 0.5 for iid in internship_ids}
        
        # Get recommendations based on similar users' preferences
        user_item_matrix, _, _ = self.feedback_manager.get_user_feedback_matrix()
        
        scores = {}
        for internship_id in internship_ids:
            if internship_id in user_item_matrix.columns:
                weighted_score = 0
                total_weight = 0
                
                for similar_user, similarity in user_similarities.items():
                    if similar_user in user_item_matrix.index:
                        user_rating = user_item_matrix.loc[similar_user, internship_id]
                        if user_rating > 0:
                            weighted_score += similarity * user_rating
                            total_weight += abs(similarity)
                
                if total_weight > 0:
                    scores[internship_id] = weighted_score / total_weight / 5.0  # Normalize to 0-1
                else:
                    scores[internship_id] = 0.5
            else:
                scores[internship_id] = 0.5
        
        return scores
    
    def get_item_based_recommendations(self, user_id: str, internship_ids: List[str]) -> Dict[str, float]:
        """Get item-based collaborative filtering scores."""
        if not self.is_trained:
            return {iid: 0.5 for iid in internship_ids}
        
        # Get user's previous interactions
        user_item_matrix, _, _ = self.feedback_manager.get_user_feedback_matrix()
        
        if user_id not in user_item_matrix.index:
            return {iid: 0.5 for iid in internship_ids}
        
        user_ratings = user_item_matrix.loc[user_id]
        liked_items = user_ratings[user_ratings >= 4].index.tolist()
        
        scores = {}
        for internship_id in internship_ids:
            if internship_id in self.item_similarity_df.index:
                # Find items similar to user's liked items
                item_score = 0
                count = 0
                
                for liked_item in liked_items:
                    if liked_item in self.item_similarity_df.columns:
                        similarity = self.item_similarity_df.loc[internship_id, liked_item]
                        item_score += similarity
                        count += 1
                
                scores[internship_id] = (item_score / count) if count > 0 else 0.5
            else:
                scores[internship_id] = 0.5
        
        return scores

# ----------------------------
# Utility Functions (same as before)
# ----------------------------
@lru_cache(maxsize=1024)
def normalize_gender_input(gender_input: Optional[str]) -> str:
    """Normalize gender input with caching."""
    if not gender_input:
        return 'any'
    gender_str = str(gender_input).lower().strip()
    if gender_str in ['male', 'm', 'boy', 'men']:
        return 'male'
    elif gender_str in ['female', 'f', 'girl', 'women', 'w']:
        return 'female'
    return 'any'

@lru_cache(maxsize=2048)
def preprocess_text(text: Optional[str]) -> str:
    """Optimized text preprocessing with caching."""
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s,]', '', text)
    return text.strip()

@lru_cache(maxsize=512)
def normalize_skill(skill: str) -> str:
    """Normalize skill names with caching."""
    skill = skill.lower().strip()
    for canonical, variants in Config.SKILL_MAPPING.items():
        if skill in variants:
            return canonical
    return skill

def extract_skills(skills_text: Optional[str]) -> List[str]:
    """Optimized skill extraction."""
    if pd.isna(skills_text) or not skills_text:
        return []

    skills = re.split(r'[,;|&\n]', str(skills_text))
    
    cleaned_skills = set()
    prefix_pattern = re.compile(r'^(knowledge of|experience in|familiar with|proficiency in)')
    
    for skill in skills:
        skill = skill.strip().lower()
        skill = prefix_pattern.sub('', skill).strip()
        skill = re.sub(r'\s+', ' ', skill).strip()
        
        if skill and len(skill) > 1:
            normalized = normalize_skill(skill)
            cleaned_skills.add(normalized)
    
    return list(cleaned_skills)

@lru_cache(maxsize=512)
def is_paid(stipend: Union[int, float, str]) -> bool:
    """Optimized payment determination with caching."""
    if isinstance(stipend, (int, float)):
        return stipend > 0
    
    stipend_str = str(stipend).lower().strip()
    if not stipend_str or stipend_str in ['unpaid', '0', '0/month', '0/year', 'performance based', 'nan']:
        return False
    
    number_match = re.search(r'(\d+)', stipend_str)
    return bool(number_match and int(number_match.group(1)) > 0)

@lru_cache(maxsize=512)
def determine_location_type(location: Optional[str]) -> str:
    """Optimized location type determination with caching."""
    if not location or pd.isna(location):
        return 'remote'
    
    location_str = str(location).lower().strip()
    remote_keywords = ['work from home', 'remote', 'wfh', 'anywhere']
    
    return 'remote' if any(keyword in location_str for keyword in remote_keywords) else 'onsite'

# ----------------------------
# Data Processing (FIXED: Use correct column names)
# ----------------------------
class DataProcessor:
    """Separate class for data processing operations."""
    
    @staticmethod
    def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimized data preparation."""
        df = df.copy()
        
        # Add internship ID if not present
        if 'internship_id' not in df.columns:
            df['internship_id'] = df.index.astype(str)
        
        # Process title
        title_col = next((col for col in ['internship_title', 'title'] if col in df.columns), None)
        df['processed_title'] = df[title_col].apply(preprocess_text) if title_col else ''
        
        # FIXED: Process skills with correct column priority - use 'Skills' first
        skills_column = next((col for col in ['Skills', 'skills'] if col in df.columns), None)
        if skills_column:
            df['processed_skills'] = df[skills_column].apply(preprocess_text)
            df['skill_list'] = df[skills_column].apply(extract_skills)
        else:
            df['processed_skills'] = ''
            df['skill_list'] = [[] for _ in range(len(df))]
        
        # Process location with vectorized operations
        if 'location' in df.columns:
            df['processed_location'] = df['location'].apply(preprocess_text)
            df['location_type'] = df['location'].apply(determine_location_type)
            df['display_location'] = df['location'].apply(
                lambda x: 'Work From Home' if determine_location_type(x) == 'remote' 
                else str(x) if x else 'Not Specified'
            )
        else:
            df['processed_location'] = 'work from home'
            df['location_type'] = 'remote'
            df['display_location'] = 'Work From Home'
        
        # Process gender and payment
        df['processed_gender'] = df.get('gender', 'any').apply(normalize_gender_input)
        df['is_paid'] = df.get('stipend', 0).apply(is_paid)
        
        return df

# ----------------------------
# Embedding Manager (same as before)
# ----------------------------
class EmbeddingManager:
    """Separate class for embedding management."""
    
    def __init__(self, cache_dir: str = Config.CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.model = None
    
    def get_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(Config.MODEL_NAME)
        return self.model
    
    def get_cache_path(self, data_size: int) -> Path:
        """Generate cache file path."""
        cache_hash = hashlib.md5(str(data_size).encode()).hexdigest()[:8]
        return self.cache_dir / f"embeddings_{data_size}_{cache_hash}.pkl"
    
    def load_embeddings(self, data_size: int) -> Optional[np.ndarray]:
        """Load embeddings from cache."""
        cache_path = self.get_cache_path(data_size)
        if cache_path.exists():
            try:
                return joblib.load(cache_path)
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
        return None
    
    def save_embeddings(self, embeddings: np.ndarray, data_size: int) -> None:
        """Save embeddings to cache."""
        cache_path = self.get_cache_path(data_size)
        try:
            joblib.dump(embeddings, cache_path, compress=3)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")
    
    def compute_embeddings_batch(self, skill_lists: List[List[str]], batch_size: int = 64) -> np.ndarray:
        """Compute embeddings in batches for better memory efficiency."""
        model = self.get_model()
        embeddings = []
        
        texts = []
        for skills in skill_lists:
            if skills:
                texts.append(", ".join(skills))
            else:
                texts.append("")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        embedding_dim = model.get_sentence_embedding_dimension()
        for i, skills in enumerate(skill_lists):
            if not skills:
                embeddings[i] = np.zeros(embedding_dim)
        
        return np.array(embeddings)

# ----------------------------
# Enhanced Recommender System
# ----------------------------
class OptimizedInternshipRecommender:
    """Enhanced recommendation system with collaborative filtering."""
    
    def __init__(self, dataframe: pd.DataFrame, cache_dir: str = Config.CACHE_DIR):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing enhanced recommender with collaborative filtering...")
        
        self.df = DataProcessor.prepare_dataframe(dataframe)
        self.embedding_manager = EmbeddingManager(cache_dir)
        self.internship_embeddings = self._load_or_compute_embeddings()
        
        # Initialize collaborative filtering components
        self.feedback_manager = FeedbackManager()
        self.collaborative_filter = CollaborativeFilter(self.feedback_manager)
        
        # Pre-compute arrays for faster access
        self.location_types = self.df['location_type'].values
        self.processed_genders = self.df['processed_gender'].values
        self.is_paid_array = self.df['is_paid'].values
        self.internship_ids = self.df['internship_id'].values
        
        # Train collaborative filtering model
        self.train_collaborative_model()
        
        self.logger.info("Initialization complete!")
    
    def _load_or_compute_embeddings(self) -> np.ndarray:
        """Load or compute embeddings with improved caching."""
        data_size = len(self.df)
        
        embeddings = self.embedding_manager.load_embeddings(data_size)
        if embeddings is not None:
            self.logger.info(f"Loaded {len(embeddings)} cached embeddings")
            return embeddings
        
        self.logger.info("Computing fresh embeddings...")
        skill_lists = self.df['skill_list'].tolist()
        embeddings = self.embedding_manager.compute_embeddings_batch(skill_lists)
        
        self.embedding_manager.save_embeddings(embeddings, data_size)
        self.logger.info(f"Computed and cached {len(embeddings)} embeddings")
        
        return embeddings
    
    def train_collaborative_model(self):
        """Train the collaborative filtering model."""
        self.collaborative_filter.train_model()
    
    def add_user_feedback(self, user_id: str, internship_id: str, feedback_type: str, 
                         user_profile: Optional[Dict] = None, rating: Optional[int] = None):
        """Add user feedback and retrain model if necessary."""
        if user_profile:
            self.feedback_manager.add_user(user_id, user_profile)
        
        self.feedback_manager.add_feedback(user_id, internship_id, feedback_type, rating)
        
        # Retrain model periodically (you might want to implement more sophisticated logic)
        # For now, we'll retrain every time (in production, you'd want to batch this)
        self.collaborative_filter.train_model()
    
    def calculate_skills_similarity(self, user_skills: str) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized skill similarity calculation."""
        if not user_skills or not user_skills.strip():
            return np.zeros(len(self.df)), np.zeros(len(self.df))
        
        user_skills_list = extract_skills(user_skills)
        if not user_skills_list:
            return np.zeros(len(self.df)), np.zeros(len(self.df))
        
        user_skills_set = set(user_skills_list)
        
        # Vectorized exact matching
        exact_scores = []
        for skill_list in self.df['skill_list']:
            internship_skills = set(skill_list)
            common_skills = len(user_skills_set.intersection(internship_skills))
            
            if user_skills_set:
                exact_score = common_skills / len(user_skills_set)
                if common_skills > 0:
                    exact_score += common_skills * 0.1
                exact_scores.append(min(exact_score, 1.0))
            else:
                exact_scores.append(0.0)
        
        # Semantic similarity
        user_embedding = self.embedding_manager.get_model().encode([", ".join(user_skills_list)])[0]
        semantic_similarities = cosine_similarity(
            user_embedding.reshape(1, -1), 
            self.internship_embeddings
        )[0]
        
        return np.array(exact_scores), semantic_similarities
    
    def calculate_location_similarity(self, user_location: str) -> np.ndarray:
        """Optimized location similarity calculation."""
        if not user_location or not user_location.strip():
            return np.where(self.location_types == 'remote', 0.8, 0.6)
        
        user_location_processed = preprocess_text(user_location)
        user_location_type = determine_location_type(user_location)
        
        scores = np.where(
            (user_location_type == 'remote') & (self.location_types == 'remote'), 1.0,
            np.where(
                (user_location_type == 'onsite') & (self.location_types == 'onsite'), 0.8,
                np.where(
                    (user_location_type == 'onsite') & (self.location_types == 'remote'), 0.7,
                    0.2
                )
            )
        )
        
        return scores
    
    def calculate_collaborative_scores(self, user_id: str) -> np.ndarray:
        """Calculate collaborative filtering scores."""
        internship_ids = self.internship_ids.tolist()
        
        if not self.collaborative_filter.is_trained:
            # If CF is not trained, return popularity scores
            popularity_scores = self.feedback_manager.get_internship_popularity_scores()
            scores = [popularity_scores.get(iid, 0.5) for iid in internship_ids]
            return np.array(scores)
        
        # Get user-based CF scores
        user_based_scores = self.collaborative_filter.get_user_based_recommendations(
            user_id, internship_ids
        )
        
        # Get item-based CF scores
        item_based_scores = self.collaborative_filter.get_item_based_recommendations(
            user_id, internship_ids
        )
        
        # Get popularity scores
        popularity_scores = self.feedback_manager.get_internship_popularity_scores()
        
        # Combine different CF approaches
        combined_scores = []
        for iid in internship_ids:
            user_score = user_based_scores.get(iid, 0.5)
            item_score = item_based_scores.get(iid, 0.5)
            popularity_score = popularity_scores.get(iid, 0.5)
            
            # Weighted combination of CF approaches
            combined_score = (0.5 * user_score + 0.3 * item_score + 0.2 * popularity_score)
            combined_scores.append(combined_score)
        
        return np.array(combined_scores)
    
    def recommend_internships(self, 
                            user_location: str = "", 
                            user_skills: str = "", 
                            user_gender: str = "any",
                            user_payment_preference: str = "any", 
                            user_id: str = None,
                            top_n: int = Config.DEFAULT_TOP_N, 
                            min_score: float = Config.MIN_SCORE_THRESHOLD) -> pd.DataFrame:
        """Enhanced recommendation with collaborative filtering."""
        self.logger.info("Calculating recommendations with collaborative filtering...")
        
        # Calculate content-based similarities
        exact_skills_sim, semantic_skills_sim = self.calculate_skills_similarity(user_skills)
        location_sim = self.calculate_location_similarity(user_location)
        
        # Calculate collaborative filtering scores
        if user_id:
            collaborative_scores = self.calculate_collaborative_scores(user_id)
        else:
            # If no user_id provided, use popularity scores only
            popularity_scores = self.feedback_manager.get_internship_popularity_scores()
            collaborative_scores = np.array([
                popularity_scores.get(iid, 0.5) for iid in self.internship_ids
            ])
        
        # Vectorized matching for other factors
        user_gender_processed = normalize_gender_input(user_gender)
        gender_match = np.where(
            (self.processed_genders == 'any') | 
            (user_gender_processed == 'any') | 
            (self.processed_genders == user_gender_processed), 1.0, 0.8
        )
        
        # Payment matching
        if user_payment_preference.lower() == 'paid':
            payment_match = np.where(self.is_paid_array, 1.0, 0.3)
        elif user_payment_preference.lower() == 'unpaid':
            payment_match = np.where(self.is_paid_array, 0.3, 1.0)
        else:
            payment_match = np.ones(len(self.df))
        
        # Calculate total scores with collaborative filtering
        total_scores = (Config.WEIGHTS['skills_exact'] * exact_skills_sim +
                       Config.WEIGHTS['skills_semantic'] * semantic_skills_sim +
                       Config.WEIGHTS['location'] * location_sim +
                       Config.WEIGHTS['collaborative'] * collaborative_scores +
                       Config.WEIGHTS['gender'] * gender_match +
                       Config.WEIGHTS['payment'] * payment_match)
        
        # Create result efficiently
        result_df = self.df.copy()
        result_df['match_score'] = total_scores
        result_df['skills_exact_similarity'] = exact_skills_sim
        result_df['skills_semantic_similarity'] = semantic_skills_sim
        result_df['location_similarity'] = location_sim
        result_df['collaborative_score'] = collaborative_scores
        result_df['work_mode'] = np.where(
            result_df['display_location'] == 'Work From Home', 'remote', 'onsite'
        )
        
        # Filter and sort efficiently
        mask = total_scores >= min_score
        filtered_df = result_df[mask]
        recommendations = filtered_df.nlargest(top_n, 'match_score')
        
        self.logger.info(f"Found {len(recommendations)} recommendations using collaborative filtering")
        return recommendations
    
    def get_user_statistics(self, user_id: str) -> Dict:
        """Get user interaction statistics."""
        conn = sqlite3.connect(Config.USER_FEEDBACK_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_interactions,
                SUM(CASE WHEN feedback_type = 'upvote' THEN 1 ELSE 0 END) as upvotes,
                SUM(CASE WHEN feedback_type = 'downvote' THEN 1 ELSE 0 END) as downvotes,
                SUM(CASE WHEN feedback_type = 'apply' THEN 1 ELSE 0 END) as applications,
                AVG(rating) as avg_rating
            FROM feedback 
            WHERE user_id = ?
        ''', (user_id,))
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'total_interactions': stats[0] or 0,
            'upvotes': stats[1] or 0,
            'downvotes': stats[2] or 0,
            'applications': stats[3] or 0,
            'avg_rating': stats[4] or 0
        }
    
    def get_trending_internships(self, days: int = 7, top_n: int = 10) -> pd.DataFrame:
        """Get trending internships based on recent feedback."""
        conn = sqlite3.connect(Config.USER_FEEDBACK_DB)
        
        query = '''
            SELECT 
                internship_id,
                COUNT(*) as recent_interactions,
                SUM(CASE WHEN feedback_type = 'upvote' THEN 1 ELSE 0 END) as recent_upvotes,
                SUM(CASE WHEN feedback_type = 'apply' THEN 1 ELSE 0 END) as recent_applications,
                AVG(rating) as recent_avg_rating
            FROM feedback 
            WHERE created_at >= datetime('now', '-{} days')
            GROUP BY internship_id
            ORDER BY (recent_upvotes + recent_applications * 2) DESC
            LIMIT ?
        '''.format(days)
        
        trending_df = pd.read_sql_query(query, conn, params=[top_n])
        conn.close()
        
        if trending_df.empty:
            return pd.DataFrame()
        
        # Merge with internship details
        trending_internships = self.df[
            self.df['internship_id'].isin(trending_df['internship_id'])
        ].merge(trending_df, on='internship_id')
        
        return trending_internships.sort_values('recent_interactions', ascending=False)
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings and reset collaborative model."""
        import shutil
        cache_path = Path(Config.CACHE_DIR)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            cache_path.mkdir(exist_ok=True)
        
        # Reset collaborative filtering
        self.collaborative_filter = CollaborativeFilter(self.feedback_manager)
        self.logger.info("Cache and collaborative model cleared!")

def display_recommendations(recommendations: pd.DataFrame, show_cf_scores: bool = True) -> None:
    """Display recommendations with collaborative filtering scores."""
    if recommendations.empty:
        print("No matching internships found. Try broader criteria.")
        return
    
    print(f"\nTop {len(recommendations)} Recommendations:")
    print("=" * 80)
    
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        title = row.get('internship_title') or row.get('title', 'Unknown Title')
        company = row.get('company_name') or row.get('company', 'Unknown Company')
        location = row.get('display_location', 'N/A')
        stipend = row.get('stipend', 'N/A')
        duration = row.get('duration', 'N/A')
        skills = (row.get('Skills') or row.get('skills'))
        payment_status = "Paid" if row['is_paid'] else "Unpaid"
        
        print(f"\n{i}. {title}")
        print(f"   Company: {company}")
        print(f"   Location: {location}")
        print(f"   Stipend: {stipend} ({payment_status})")
        print(f"   Duration: {duration}")
        print(f"   Required Skills: {skills}")
        print(f"   Match Score: {row['match_score']:.3f}")
        
        if show_cf_scores:
            print(f"   Score Breakdown:")
            print(f"     - Skills (Exact): {row['skills_exact_similarity']:.2f}")
            print(f"     - Skills (Semantic): {row['skills_semantic_similarity']:.2f}")
            print(f"     - Location: {row['location_similarity']:.2f}")
            print(f"     - Collaborative: {row['collaborative_score']:.2f}")

def interactive_feedback_session(recommender: OptimizedInternshipRecommender, 
                               user_id: str, recommendations: pd.DataFrame) -> None:
    """Interactive session to collect user feedback."""
    print("\n" + "="*50)
    print("FEEDBACK COLLECTION")
    print("Help us improve recommendations by providing feedback!")
    print("Commands: upvote, downvote, apply, skip, rate [1-5], done")
    print("="*50)
    
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        internship_id = row['internship_id']
        title = row.get('internship_title') or row.get('title', 'Unknown Title')
        company = row.get('company_name') or row.get('company', 'Unknown Company')
        
        print(f"\n{i}. {title} at {company}")
        feedback = input("Your feedback (upvote/downvote/apply/skip/rate [1-5]/done): ").strip().lower()
        
        if feedback == 'done':
            break
        elif feedback in ['upvote', 'downvote', 'apply', 'skip']:
            recommender.add_user_feedback(user_id, internship_id, feedback)
            print(f"✓ Recorded {feedback} for {title}")
        elif feedback.startswith('rate'):
            try:
                rating = int(feedback.split()[1])
                if 1 <= rating <= 5:
                    recommender.add_user_feedback(user_id, internship_id, 'rate', rating=rating)
                    print(f"✓ Recorded rating {rating} for {title}")
                else:
                    print("Rating must be between 1-5")
            except (IndexError, ValueError):
                print("Invalid rating format. Use: rate [1-5]")
        else:
            print("Invalid command. Try: upvote, downvote, apply, skip, rate [1-5], or done")
    
    print("\nThank you for your feedback! This will help improve future recommendations.")

def main():
    """Enhanced main function with collaborative filtering."""
    print("ENHANCED INTERNSHIP RECOMMENDATION SYSTEM")
    print("With Collaborative Filtering Based on User Feedback")
    print("=" * 60)
    
    try:
        print("Loading CSV file...")
        df = pd.read_csv("final_internship.csv")
        print(f"Loaded {len(df)} internships from dataset")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("Error: final_internship.csv not found. Please ensure the file is in the same directory.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    try:
        recommender = OptimizedInternshipRecommender(df)
    except Exception as e:
        print(f"Error initializing recommender: {e}")
        return
    
    print("\n" + "="*60)
    
    # User identification
    user_id = input("Enter your user ID (or press Enter for anonymous): ").strip()
    if not user_id:
        user_id = f"anonymous_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Using anonymous ID: {user_id}")
    
    # Check if user exists and show stats
    if user_id:
        user_stats = recommender.get_user_statistics(user_id)
        if user_stats['total_interactions'] > 0:
            print(f"\nWelcome back! Your stats:")
            print(f"  Total interactions: {user_stats['total_interactions']}")
            print(f"  Upvotes: {user_stats['upvotes']}, Downvotes: {user_stats['downvotes']}")
            print(f"  Applications: {user_stats['applications']}")
            if user_stats['avg_rating']:
                print(f"  Average rating: {user_stats['avg_rating']:.1f}/5")
    
    # Get user preferences
    location = input("Enter preferred location (or leave empty for any): ").strip()
    skills = input("Enter your skills (comma separated): ").strip()
    gender = input("Enter gender preference (male/female/any): ").strip()
    payment_preference = input("Enter payment preference (paid/unpaid/any): ").strip()
    
    # Create user profile
    user_profile = {
        'skills': skills,
        'location': location,
        'gender': gender,
        'payment_preference': payment_preference,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        print("\nSearching for recommendations...")
        recommendations = recommender.recommend_internships(
            user_location=location,
            user_skills=skills,
            user_gender=gender,
            user_payment_preference=payment_preference,
            user_id=user_id,
            top_n=50
        )
        
        display_recommendations(recommendations, show_cf_scores=True)
        
        # Offer feedback collection
        if not recommendations.empty:
            collect_feedback = input("\nWould you like to provide feedback to improve recommendations? (y/n): ").strip().lower()
            if collect_feedback == 'y':
                interactive_feedback_session(recommender, user_id, recommendations.head(10))
        
        # Show trending internships
        show_trending = input("\nWould you like to see trending internships? (y/n): ").strip().lower()
        if show_trending == 'y':
            trending = recommender.get_trending_internships(days=7, top_n=5)
            if not trending.empty:
                print("\n" + "="*50)
                print("TRENDING INTERNSHIPS (Last 7 days)")
                print("="*50)
                for i, (_, row) in enumerate(trending.iterrows(), 1):
                    title = row.get('internship_title') or row.get('title', 'Unknown Title')
                    company = row.get('company_name') or row.get('company', 'Unknown Company')
                    interactions = row['recent_interactions']
                    upvotes = row['recent_upvotes']
                    applications = row['recent_applications']
                    
                    print(f"{i}. {title} at {company}")
                    print(f"   Recent activity: {interactions} interactions, {upvotes} upvotes, {applications} applications")
            else:
                print("No trending internships found.")
        
    except Exception as e:
        print(f"Error during recommendation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()