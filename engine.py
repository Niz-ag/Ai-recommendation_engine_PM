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

def extract_skills(skills_text):
    """Extract and clean individual skills from text."""
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
        skill = re.sub(r'\s+', ' ', skill)  # Normalize whitespace
        if skill and len(skill) > 1:  # Filter out empty or single character skills
            cleaned_skills.append(skill)

    return cleaned_skills

def is_paid(stipend):
    """Determine if an internship is paid based on the stipend value."""
    if isinstance(stipend, (int, float)):
        return True
    stipend_str = str(stipend).lower().strip()
    # Check for keywords that indicate it's not paid, including 0
    if not stipend_str or stipend_str in ['unpaid', '0', '0/month', '0/year']:
        return False
    # If the string contains a number greater than 0, assume it's paid
    if re.search(r'\d+', stipend_str):
        number_match = re.search(r'(\d+)', stipend_str)
        if number_match and int(number_match.group(1)) > 0:
            return True
    return False

def determine_display_location(location, work_mode):
    """Determine the display location based on work mode."""
    work_mode_lower = str(work_mode).lower() if work_mode else 'remote'
    
    if work_mode_lower == 'onsite':
        return location if location else 'Office Location'
    elif work_mode_lower == 'remote':
        return 'Work from Home'
    elif work_mode_lower == 'hybrid':
        base_location = location if location else 'Office'
        return f"{base_location} / Work from Home"
    else:
        return location if location else 'Remote'

class EnhancedInternshipRecommender:
    def __init__(self, dataframe, cache_dir="embeddings_cache"):
        print("Initializing recommender...")
        self.df = dataframe.copy()
        self.cache_dir = cache_dir

        # Create cache directory
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

        # Departments (replacing titles)
        # if 'department' in self.df.columns:
        #     self.df['processed_department'] = self.df['department'].apply(preprocess_text)
        # elif 'internship_department' in self.df.columns:
        #     self.df['processed_department'] = self.df['internship_department'].apply(preprocess_text)
        # else:
        #     self.df['processed_department'] = ''

        # Keep titles for display but don't use for matching
        if 'internship_title' in self.df.columns:
            self.df['processed_title'] = self.df['internship_title'].apply(preprocess_text)
        elif 'title' in self.df.columns:
            self.df['processed_title'] = self.df['title'].apply(preprocess_text)
        else:
            self.df['processed_title'] = ''

        # Skills - Enhanced processing
        skills_column = None
        for col in ['required_skills', 'skills', 'Skills']:
            if col in self.df.columns:
                skills_column = col
                break

        if skills_column:
            print(f"Found skills column: {skills_column}")
            self.df['processed_skills'] = self.df[skills_column].apply(preprocess_text)
            self.df['skill_list'] = self.df[skills_column].apply(extract_skills)
        else:
            print("No skills column found, using empty skills")
            self.df['processed_skills'] = ''
            self.df['skill_list'] = [[] for _ in range(len(self.df))]

        # Work mode and location
        if 'work_mode' in self.df.columns:
            self.df['processed_work_mode'] = self.df['work_mode'].apply(preprocess_text)
        else:
            self.df['processed_work_mode'] = 'remote'

        # Update location display based on work mode
        if 'location' in self.df.columns:
            self.df['processed_location'] = self.df['location'].apply(preprocess_text)
            self.df['display_location'] = self.df.apply(
                lambda row: determine_display_location(row['location'], row.get('work_mode')), axis=1
            )
        else:
            self.df['processed_location'] = ''
            self.df['display_location'] = 'Work from Home'

        # Gender
        if 'gender' in self.df.columns:
            self.df['processed_gender'] = self.df['gender'].apply(normalize_gender_input)
        else:
            self.df['processed_gender'] = 'any'

        # Payment status
        if 'stipend' in self.df.columns:
            self.df['is_paid'] = self.df['stipend'].apply(is_paid)
        else:
            # Assume all are paid if no stipend column exists
            self.df['is_paid'] = True

        print("Data preprocessing complete")

    def _get_cache_path(self):
        """Generate cache file path."""
        # Simple cache name based on dataset size
        cache_name = f"embeddings_{len(self.df)}_{hashlib.md5(str(len(self.df)).encode()).hexdigest()[:8]}.pkl"
        return os.path.join(self.cache_dir, cache_name)

    def _load_or_compute_embeddings(self):
        """Load from cache or compute fresh embeddings."""
        cache_path = self._get_cache_path()

        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                print("Loading cached embeddings...")
                with open(cache_path, 'rb') as f:
                    self.internship_skill_embeddings = pickle.load(f)
                print(f"Loaded {len(self.internship_skill_embeddings)} cached embeddings")
                return
            except Exception as e:
                print(f"Cache load failed: {e}")

        # Compute fresh embeddings
        print("Computing fresh embeddings...")
        self._compute_embeddings()

        # Save to cache
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
        total = len(self.df)

        print(f"Computing embeddings for {total} internships...")

        for idx, row in self.df.iterrows():
            if idx % 500 == 0:
                print(f"Progress: {idx}/{total} ({100*idx/total:.1f}%)")

            skills_list = row.get('skill_list', [])
            if skills_list:
                skills_text = ", ".join(skills_list)
                embedding = self.semantic_model.encode([skills_text])[0]
            else:
                # Zero embedding for no skills
                embedding = np.zeros(self.semantic_model.get_sentence_embedding_dimension())

            embeddings.append(embedding)

        self.internship_skill_embeddings = np.array(embeddings)
        print(f"Computed {len(embeddings)} embeddings")

    def calculate_semantic_skills_similarity(self, user_skills_text):
        """Calculate semantic similarity between user skills and internship requirements."""
        if not user_skills_text or not user_skills_text.strip():
            return np.zeros(len(self.df))

        # Extract and process user skills
        user_skills_list = extract_skills(user_skills_text)
        if not user_skills_list:
            return np.zeros(len(self.df))

        # Create user skills embedding
        user_skills_combined = ", ".join(user_skills_list)
        user_embedding = self.semantic_model.encode([user_skills_combined])[0]

        # Calculate cosine similarity with all internship embeddings
        similarities = []
        for internship_embedding in self.internship_skill_embeddings:
            if np.any(internship_embedding):  # Check if not zero vector
                similarity = cosine_similarity(
                    user_embedding.reshape(1, -1),
                    internship_embedding.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)
            else:
                similarities.append(0.0)

        return np.array(similarities)

    def calculate_location_similarity(self, user_location, user_work_mode):
        """Enhanced location matching based on work mode."""
        user_work_mode_lower = str(user_work_mode).lower() if user_work_mode else 'remote'
        
        if user_work_mode_lower == 'remote':
            # For remote work, location doesn't matter much
            return np.ones(len(self.df)) * 0.8  # Give decent score to all

        if not user_location or not user_location.strip():
            return np.zeros(len(self.df))

        user_location_processed = preprocess_text(user_location)
        user_parts = set(user_location_processed.split())

        location_scores = []
        for _, row in self.df.iterrows():
            internship_work_mode = str(row.get('processed_work_mode', 'remote')).lower()
            internship_location = row.get('processed_location', '')

            # If internship is remote, good match for any user preference
            if internship_work_mode == 'remote':
                location_scores.append(0.9)
                continue

            # If user wants onsite/hybrid, check location match
            if user_work_mode_lower in ['onsite', 'hybrid']:
                if not internship_location:
                    location_scores.append(0.0)
                    continue

                internship_parts = set(internship_location.split())

                # Exact match
                if user_location_processed == internship_location:
                    location_scores.append(1.0)
                # Partial match based on common words
                elif user_parts & internship_parts:
                    # Calculate Jaccard similarity
                    intersection = len(user_parts & internship_parts)
                    union = len(user_parts | internship_parts)
                    similarity = intersection / union if union > 0 else 0.0
                    location_scores.append(similarity)
                else:
                    location_scores.append(0.0)
            else:
                location_scores.append(0.0)

        return np.array(location_scores)

    def calculate_payment_match(self, user_payment_preference):
        """Calculate a score based on the user's payment preference."""
        payment_scores = []
        user_pref = user_payment_preference.lower().strip()

        for is_paid_internship in self.df['is_paid']:
            if user_pref == 'any':
                payment_scores.append(1.0)
            elif user_pref == 'paid':
                payment_scores.append(1.0 if is_paid_internship else 0.0)
            elif user_pref == 'unpaid':
                payment_scores.append(1.0 if not is_paid_internship else 0.0)
            else:
                payment_scores.append(0.0)
        return np.array(payment_scores)

    def recommend_internships(self, user_location="", user_work_mode="hybrid", 
                              user_skills="", user_gender="any", user_payment_preference="any",
                              top_n=25, weights=None, min_score=0.0):
        """Get personalized internship recommendations."""
        print("Calculating recommendations...")

        if weights is None:
            # Focus on skills and location as per requirements
            weights = {'skills': 0.7, 'location': 0.3, 'gender': 0.0, 'payment': 0.0}

        user_gender_processed = normalize_gender_input(user_gender)

        # Semantic skills similarity (primary factor)
        skills_similarity = self.calculate_semantic_skills_similarity(user_skills)

        # Location matching with work mode consideration
        location_similarity = self.calculate_location_similarity(user_location, user_work_mode)

        # Gender compatibility (not used in current weights)
        gender_match = np.array([
            1.0 if g == 'any' or user_gender_processed == 'any' else
            1.0 if g == user_gender_processed else
            0.8
            for g in self.df['processed_gender']
        ])

        # Payment preference match (not used in current weights)
        payment_match = self.calculate_payment_match(user_payment_preference)

        # Combine all scores
        total_scores = (weights['skills'] * skills_similarity +
                        weights['location'] * location_similarity +
                        weights['gender'] * gender_match +
                        weights['payment'] * payment_match)

        # Add scores to dataframe copy
        result_df = self.df.copy()
        result_df['match_score'] = total_scores
        result_df['skills_similarity'] = skills_similarity
        result_df['location_similarity'] = location_similarity
        result_df['gender_compatibility'] = gender_match
        result_df['payment_compatibility'] = payment_match

        # Filter and sort
        filtered_df = result_df[result_df['match_score'] >= min_score]
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
# # Main CLI
# # ----------------------------
# def main():
#     print("ENHANCED INTERNSHIP RECOMMENDATION SYSTEM")
#     print("=" * 50)

#     try:
#         print("Loading CSV file...")
#         df = pd.read_csv("final_internship.csv")
#         print(f"Loaded {len(df)} internships from dataset")
#         print(f"Columns: {list(df.columns)}")
#     except FileNotFoundError:
#         print("Error: final_internship.csv not found.")
#         return
#     except Exception as e:
#         print(f"Error loading CSV: {e}")
#         return

#     # Clear cache by default
#     clear_cache = False

#     # Initialize recommender
#     try:
#         recommender = EnhancedInternshipRecommender(df)
#         if clear_cache:
#             recommender.clear_cache()
#             recommender = EnhancedInternshipRecommender(df)
#     except Exception as e:
#         print(f"Error initializing recommender: {e}")
#         return

#     # Get user input for search criteria
#     print("\n" + "="*50)
#     department = input("Enter desired internship department: ").strip()
#     location = input("Enter preferred location: ").strip()
#     work_mode = input("Enter work mode preference (onsite/remote/hybrid): ").strip()
#     skills = input("Enter your skills (comma separated): ").strip()
#     gender = input("Enter gender (male/female/any): ").strip()
#     payment_preference = input("Enter payment preference (paid/unpaid/any): ").strip()

#     try:
#         print("\nSearching for recommendations...")
#         recommendations = recommender.recommend_internships(
#             department, location, work_mode, skills, gender, payment_preference, top_n=50
#         )

#         if recommendations.empty:
#             print("No matching internships found. Try broader criteria.")
#         else:
#             print(f"\nTop {len(recommendations)} Recommendations:")
#             print("=" * 60)

#             for i, (_, row) in enumerate(recommendations.iterrows(), 1):
#                 # Handle different column names flexibly
#                 title_val = row.get('internship_title') or row.get('title', 'Unknown Title')
#                 department_val = row.get('department') or row.get('internship_department', 'Unknown Department')
#                 company_val = row.get('company_name') or row.get('company', 'Unknown Company')
#                 location_val = row.get('display_location', 'N/A')
#                 stipend_val = row.get('stipend', 'N/A')
#                 duration_val = row.get('duration', 'N/A')
#                 gender_val = row.get('gender', 'Any')
#                 work_mode_val = row.get('work_mode', 'Remote')

#                 # Get skills from any possible column
#                 skills_val = (row.get('required_skills') or
#                              row.get('skills') or
#                              row.get('Skills', 'N/A'))

#                 payment_status = "Paid" if row['is_paid'] else "Unpaid"

#                 print(f"\n{i}. {title_val}")
#                 # print(f"   Department: {department_val}")
#                 print(f"   Company: {company_val}")
#                 print(f"   Location: {location_val}")
#                 print(f"   Work Mode: {work_mode_val}")
#                 print(f"   Stipend: {stipend_val} ({payment_status})")
#                 print(f"   Duration: {duration_val}")
#                 print(f"   Gender: {gender_val}")
#                 print(f"   Skills: {skills_val}")
#                 print(f"   Match Score: {row['match_score']:.3f}")
#                 print(f"   Breakdown: Skills={row['skills_similarity']:.2f}, "
#                       f"Location={row['location_similarity']:.2f}")

#     except Exception as e:
#         print(f"Error during recommendation: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()