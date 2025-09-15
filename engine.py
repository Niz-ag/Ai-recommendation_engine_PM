import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import pickle
import os
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Helper functions
# ----------------------------
def normalize_gender_input(gender_input):
    """Normalize gender input."""
    if not gender_input:
        return 'any'
    gender_str = str(gender_input).lower().strip()
    if gender_str in ['male', 'm', 'boy', 'men']:
        return 'male'
    elif gender_str in ['female', 'f', 'girl', 'women', 'w']:
        return 'female'
    else:
        return 'any'

def preprocess_text(text):
    """Lowercase, remove punctuation, and strip text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s,]', '', text)  # Keep commas for skill separation
    return text.strip()

def normalize_skill(skill):
    """Normalize skill names for better matching."""
    skill = skill.lower().strip()

    # Skill normalization mapping
    skill_mapping = {
        'javascript': ['js', 'javascript', 'java script'],
        'python': ['python', 'py'],
        'java': ['java'],  # Keep separate from javascript
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

    # Find the canonical form of the skill
    for canonical, variants in skill_mapping.items():
        if skill in variants:
            return canonical

    return skill

def extract_skills(skills_text):
    """Extract and clean individual skills from text with better normalization."""
    if pd.isna(skills_text) or not skills_text:
        return []

    # Split by common delimiters
    skills = re.split(r'[,;|&\n]', str(skills_text))

    # Clean each skill
    cleaned_skills = []
    for skill in skills:
        skill = skill.strip().lower()
        # Remove common prefixes/suffixes
        skill = re.sub(r'^(knowledge of|experience in|familiar with|proficiency in)', '', skill).strip()
        skill = re.sub(r'\s+', ' ', skill).strip()  # Normalize whitespace

        if skill and len(skill) > 1:  # Filter out empty or single character skills
            # Normalize the skill
            normalized = normalize_skill(skill)
            cleaned_skills.append(normalized)

    return list(set(cleaned_skills))  # Remove duplicates

def is_paid(stipend):
    """Determine if an internship is paid based on the stipend value."""
    if isinstance(stipend, (int, float)):
        return stipend > 0
    stipend_str = str(stipend).lower().strip()
    # Check for keywords that indicate it's not paid
    if not stipend_str or stipend_str in ['unpaid', '0', '0/month', '0/year', 'performance based', 'nan']:
        return False
    # If the string contains a number greater than 0, assume it's paid
    if re.search(r'\d+', stipend_str):
        number_match = re.search(r'(\d+)', stipend_str)
        if number_match and int(number_match.group(1)) > 0:
            return True
    return False

def determine_location_type(location):
    """Determine if location is remote or onsite."""
    if not location or pd.isna(location):
        return 'remote'

    location_str = str(location).lower().strip()
    remote_keywords = ['work from home', 'remote', 'wfh', 'anywhere']

    for keyword in remote_keywords:
        if keyword in location_str:
            return 'remote'

    return 'onsite'

class OptimizedInternshipRecommender:
    def __init__(self, dataframe, cache_dir="embeddings_cache"):
        print("Initializing recommender...")
        self.df = dataframe.copy()
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        print("Preparing data...")
        self._prepare_data()

        print("Loading semantic model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        print("Loading/Computing embeddings...")
        self._load_or_compute_embeddings()
        print("Initialization complete!")

    def _prepare_data(self):
        """Preprocess columns for scoring."""
        print(f"Processing {len(self.df)} records...")

        # Keep titles for display but don't use for matching
        title_col = 'internship_title' if 'internship_title' in self.df.columns else 'title'
        if title_col in self.df.columns:
            self.df['processed_title'] = self.df[title_col].apply(preprocess_text)
        else:
            self.df['processed_title'] = ''

        # Skills - Enhanced processing with better normalization
        skills_column = next((col for col in ['required_skills', 'skills', 'Skills'] if col in self.df.columns), None)
        if skills_column:
            print(f"Found skills column: {skills_column}")
            self.df['processed_skills'] = self.df[skills_column].apply(preprocess_text)
            self.df['skill_list'] = self.df[skills_column].apply(extract_skills)
        else:
            print("No skills column found, using empty skills")
            self.df['processed_skills'] = ''
            self.df['skill_list'] = [[] for _ in range(len(self.df))]

        # Location processing (removed work_mode dependency)
        if 'location' in self.df.columns:
            self.df['processed_location'] = self.df['location'].apply(preprocess_text)
            self.df['location_type'] = self.df['location'].apply(determine_location_type)
            self.df['display_location'] = self.df['location'].apply(
                lambda x: 'Work From Home' if determine_location_type(x) == 'remote' else str(x) if x else 'Not Specified'
            )
        else:
            self.df['processed_location'] = 'work from home'
            self.df['location_type'] = 'remote'
            self.df['display_location'] = 'Work From Home'

        # Gender
        if 'gender' in self.df.columns:
            self.df['processed_gender'] = self.df['gender'].apply(normalize_gender_input)
        else:
            self.df['processed_gender'] = 'any'

        # Payment status
        if 'stipend' in self.df.columns:
            self.df['is_paid'] = self.df['stipend'].apply(is_paid)
        else:
            self.df['is_paid'] = True

        print("Data preprocessing complete")

    def _get_cache_path(self):
        """Generate cache file path."""
        cache_name = f"embeddings_{len(self.df)}_{hashlib.md5(str(len(self.df)).encode()).hexdigest()[:8]}.pkl"
        return os.path.join(self.cache_dir, cache_name)

    def _load_or_compute_embeddings(self):
        """Load from cache or compute fresh embeddings."""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                print("Loading cached embeddings...")
                with open(cache_path, 'rb') as f:
                    self.internship_skill_embeddings = pickle.load(f)
                print(f"Loaded {len(self.internship_skill_embeddings)} cached embeddings")
                return
            except Exception as e:
                print(f"Cache load failed: {e}")

        print("Computing fresh embeddings...")
        self._compute_embeddings()

        try:
            print("Saving to cache...")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.internship_skill_embeddings, f)
            print("Embeddings cached successfully")
        except Exception as e:
            print(f"Cache save failed: {e}")

    def _compute_embeddings(self):
        """Compute embeddings for all internships."""
        embeddings = []
        for _, row in self.df.iterrows():
            skills_list = row.get('skill_list', [])
            if skills_list:
                skills_text = ", ".join(skills_list)
                embedding = self.semantic_model.encode([skills_text])[0]
            else:
                embedding = np.zeros(self.semantic_model.get_sentence_embedding_dimension())
            embeddings.append(embedding)
        self.internship_skill_embeddings = np.array(embeddings)
        print(f"Computed {len(embeddings)} embeddings")

    def calculate_enhanced_skills_similarity(self, user_skills_text):
        """Enhanced skill matching with exact and semantic matching."""
        if not user_skills_text or not user_skills_text.strip():
            return np.zeros(len(self.df)), np.zeros(len(self.df))

        user_skills_list = extract_skills(user_skills_text)
        if not user_skills_list:
            return np.zeros(len(self.df)), np.zeros(len(self.df))

        user_skills_set = set(user_skills_list)

        # Exact keyword matching with higher weight for exact matches
        exact_scores = []
        for _, row in self.df.iterrows():
            internship_skills = set(row.get('skill_list', []))

            # Calculate exact matches
            common_skills = len(user_skills_set.intersection(internship_skills))
            total_user_skills = len(user_skills_set)

            if total_user_skills > 0:
                exact_score = common_skills / total_user_skills
                # Bonus for having more matches
                if common_skills > 0:
                    exact_score = exact_score + (common_skills * 0.1)  # Bonus for multiple matches
                exact_scores.append(min(exact_score, 1.0))  # Cap at 1.0
            else:
                exact_scores.append(0.0)

        # Semantic similarity
        user_skills_combined = ", ".join(user_skills_list)
        user_embedding = self.semantic_model.encode([user_skills_combined])[0]
        semantic_similarities = [
            cosine_similarity(user_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]
            if np.any(emb) else 0.0 for emb in self.internship_skill_embeddings
        ]

        return np.array(exact_scores), np.array(semantic_similarities)

    def calculate_location_similarity(self, user_location):
        """Simplified location matching without work_mode."""
        if not user_location or user_location.strip() == '':
            # If no preference, slightly prefer remote but don't exclude others
            return np.array([0.8 if loc_type == 'remote' else 0.6 for loc_type in self.df['location_type']])

        user_location_processed = preprocess_text(user_location)
        user_location_type = determine_location_type(user_location)

        location_scores = []
        for _, row in self.df.iterrows():
            internship_location_type = row.get('location_type', 'remote')
            internship_location = row.get('processed_location', '')

            if user_location_type == 'remote' and internship_location_type == 'remote':
                score = 1.0
            elif user_location_type == 'onsite' and internship_location_type == 'onsite':
                # Exact location match
                if user_location_processed in internship_location or internship_location in user_location_processed:
                    score = 1.0
                else:
                    score = 0.3  # Different onsite location
            elif user_location_type == 'onsite' and internship_location_type == 'remote':
                score = 0.7  # Remote is somewhat acceptable for onsite preference
            else:  # user wants remote, internship is onsite
                score = 0.2  # Lower score for location mismatch

            location_scores.append(score)

        return np.array(location_scores)

    def recommend_internships(self, user_location="", user_skills="", user_gender="any",
                             user_payment_preference="any", top_n=25, min_score=0.1, user_mode="remote"):
        """Optimized recommendation with better skill matching."""
        print("Calculating recommendations...")

        # Optimized weights focusing more on exact skill matches
        weights = {
            'skills_exact': 0.5,      # Higher weight for exact matches
            'skills_semantic': 0.2,   # Lower weight for semantic similarity
            'location': 0.3,          # Location importance
            'gender': 0.0,           # Remove gender bias
            'payment': 0.0           # Remove payment bias unless specified
        }

        user_gender_processed = normalize_gender_input(user_gender)

        # Enhanced skills similarity
        exact_skills_similarity, semantic_skills_similarity = self.calculate_enhanced_skills_similarity(user_skills)

        # Location similarity
        location_similarity = self.calculate_location_similarity(user_location)

        # Gender and payment matching (kept for compatibility)
        gender_match = np.array([1.0 if g == 'any' or user_gender_processed == 'any' or g == user_gender_processed else 0.8 for g in self.df['processed_gender']])

        # Payment matching
        if user_payment_preference.lower() == 'paid':
            payment_match = np.array([1.0 if paid else 0.3 for paid in self.df['is_paid']])
        elif user_payment_preference.lower() == 'unpaid':
            payment_match = np.array([1.0 if not paid else 0.3 for paid in self.df['is_paid']])
        else:
            payment_match = np.ones(len(self.df))

        # Calculate total scores
        total_scores = (weights['skills_exact'] * exact_skills_similarity +
                       weights['skills_semantic'] * semantic_skills_similarity +
                       weights['location'] * location_similarity +
                       weights['gender'] * gender_match +
                       weights['payment'] * payment_match)

        # Create result dataframe
        result_df = self.df.copy()
        result_df['match_score'] = total_scores
        result_df['skills_exact_similarity'] = exact_skills_similarity
        result_df['skills_semantic_similarity'] = semantic_skills_similarity
        result_df['location_similarity'] = location_similarity
        result_df['gender_compatibility'] = gender_match
        result_df['payment_compatibility'] = payment_match
        result_df['work_mode'] = result_df['display_location'].apply(
    lambda loc: 'remote' if str(loc).lower() == 'work from home' else 'onsite'

    
)
        
        


        # Filter by minimum score
        filtered_df = result_df[result_df['match_score'] >= min_score]

        # Sort by match score and return top results
        recommendations = filtered_df.sort_values('match_score', ascending=False).head(top_n)

        print(f"Found {len(recommendations)} recommendations")
        return recommendations

    def clear_cache(self):
        """Clear all cached embeddings."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print("Cache cleared!")

# ----------------------------
# Main CLI
# ----------------------------
def main():
    print("OPTIMIZED INTERNSHIP RECOMMENDATION SYSTEM")
    print("=" * 50)

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

    print("\n" + "="*50)
    location = input("Enter preferred location (or leave empty for any): ").strip()
    skills = input("Enter your skills (comma separated): ").strip()
    gender = input("Enter gender preference (male/female/any): ").strip()
    payment_preference = input("Enter payment preference (paid/unpaid/any): ").strip()

    try:
        print("\nSearching for recommendations...")
        recommendations = recommender.recommend_internships(
            user_location=location,
            user_skills=skills,
            user_gender=gender,
            user_payment_preference=payment_preference,
            top_n=25
        )

        if recommendations.empty:
            print("No matching internships found. Try broader criteria.")
        else:
            print(f"\nTop {len(recommendations)} Recommendations:")
            print("=" * 80)
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                title_val = row.get('internship_title') or row.get('title', 'Unknown Title')
                company_val = row.get('company_name') or row.get('company', 'Unknown Company')
                location_val = row.get('display_location', 'N/A')
                stipend_val = row.get('stipend', 'N/A')
                duration_val = row.get('duration', 'N/A')
                gender_val = row.get('gender', 'Any')
                skills_val = (row.get('required_skills') or row.get('skills') or row.get('Skills', 'N/A'))
                payment_status = "Paid" if row['is_paid'] else "Unpaid"

                print(f"\n{i}. {title_val}")
                print(f"   Company: {company_val}")
                print(f"   Location: {location_val}")
                print(f"   Stipend: {stipend_val} ({payment_status})")
                print(f"   Duration: {duration_val}")
                print(f"   Required Skills: {skills_val}")
                print(f"   Match Score: {row['match_score']:.3f}")
                print(f"   Skill Breakdown: Exact={row['skills_exact_similarity']:.2f}, "
                      f"Semantic={row['skills_semantic_similarity']:.2f}, "
                      f"Location={row['location_similarity']:.2f}")

    except Exception as e:
        print(f"Error during recommendation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()