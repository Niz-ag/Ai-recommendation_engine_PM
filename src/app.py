from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import sys

   
   # Add the parent directory to the Python path to import engine.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import EnhancedInternshipRecommender, determine_display_location
import traceback

app = Flask(__name__)
CORS(app)  # Allow React frontend to communicate with Flask backend

# Global variables to store the recommender instance
recommender = None
df = None

def initialize_recommender():
    """Initialize the recommendation engine with the CSV data."""
    global recommender, df
    try:
        print("Loading CSV file...")
        # Look for CSV in the root directory (parent of src)
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_internship.csv")
        if not os.path.exists(csv_path):
            # Fallback: look in the same directory as app.py
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_internship.csv")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} internships from dataset")
        print(f"Available columns: {list(df.columns)}")
        
        print("Initializing recommender...")
        recommender = EnhancedInternshipRecommender(df)
        print("Recommender initialized successfully!")
        return True
    except FileNotFoundError:
        print("Error: final_internship.csv not found.")
        return False
    except Exception as e:
        print(f"Error initializing recommender: {e}")
        traceback.print_exc()
        return False

def format_recommendation(row, index):
    """Format a single recommendation for the frontend."""
    # Handle different column names flexibly
    # department_val = row.get('department') or row.get('internship_department', 'Unknown Department')
    company_val = row.get('company_name') or row.get('company', 'Unknown Company')
    stipend_val = row.get('stipend', 'N/A')
    duration_val = row.get('duration', '3 Months')
    work_mode_val = row.get('work_mode', 'Remote')
    
    # Get skills from any possible column
    skills_val = (row.get('required_skills') or
                 row.get('skills') or
                 row.get('Skills', ''))
    
    # Get description or create a default one
    description_val = row.get('description', f"Exciting internship opportunity")
    
    # Determine display location based on work mode
    location_val = determine_display_location(row.get('location', ''), work_mode_val)
    
    # Create requirements and benefits lists
    requirements = []
    if skills_val and skills_val != 'N/A':
        # Split skills into requirements
        skill_list = [skill.strip() for skill in str(skills_val).split(',')][:3]  # Limit to 3
        requirements = skill_list
    else:
        requirements = ["Basic computer skills", "Communication skills"]
    
    benefits = ["Professional development", "Mentorship program", "Certificate of completion"]
    
    # Add specific benefits based on payment status
    if row.get('is_paid', True):
        benefits.insert(0, f"Stipend: {stipend_val}")
    else:
        benefits.insert(0, "Valuable work experience")
    
    return {
        "id": str(index),
        "company": company_val,
        "location": location_val,
        "workMode": work_mode_val,
        "duration": duration_val,
        "stipend": stipend_val,
        "matchScore": round(row.get('match_score', 0) * 100, 1),  # Convert to percentage
        "description": description_val,
        "requirements": requirements,
        "benefits": benefits,
    }

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({
        "message": "Internship Recommendation API is running!",
        "status": "active",
        "recommender_loaded": recommender is not None
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    """Generate internship recommendations based on user input."""
    global recommender
    
    if recommender is None:
        return jsonify({
            "error": "Recommendation engine not initialized",
            "recommendations": []
        }), 500
    
    try:
        data = request.get_json()
        print("Received recommendation request:", data)

        # Extract data from frontend
        age = data.get("age")
        gender = data.get("gender", "any")
        skills = data.get("skills", "")
        location = data.get("location", "")
        income = data.get("familyIncome")
        work_mode = data.get("workMode", "remote")
        duration = data.get("duration", "")
        
        # Validate required fields
        # Get recommendations from the engine
        print(f"Getting recommendations, location: {location}, work_mode: {work_mode}")
        recommendations_df = recommender.recommend_internships(
            user_location=location,
            user_work_mode=work_mode,
            user_skills=skills,
            user_gender=gender,
            user_payment_preference="any",  # Can be made configurable
            top_n=10,
            min_score=0.0
        )

        # Format recommendations for frontend
        recommendations = []
        for idx, (_, row) in enumerate(recommendations_df.iterrows()):
            formatted_rec = format_recommendation(row, idx + 1)
            recommendations.append(formatted_rec)

        print(f"Returning {len(recommendations)} recommendations")
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        print(f"Error in recommend endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error during recommendation generation",
            "recommendations": []
        }), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get statistics about the dataset."""
    global df
    
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        stats = {
            "total_internships": len(df),
            "unique_companies": df['company_name'].nunique() if 'company_name' in df.columns else 0,
            "unique_locations": df['location'].nunique() if 'location' in df.columns else 0,
            "work_modes": df['work_mode'].value_counts().to_dict() if 'work_mode' in df.columns else {},
            "paid_vs_unpaid": {
                "paid": len(df[df['is_paid'] == True]) if 'is_paid' in df.columns else 0,
                "unpaid": len(df[df['is_paid'] == False]) if 'is_paid' in df.columns else 0
            }
        }
        return jsonify(stats)
    except Exception as e:
        print(f"Error generating stats: {e}")
        return jsonify({"error": "Error generating statistics"}), 500

@app.route("/departments", methods=["GET"])
def get_departments():
    """Get list of available departments."""
    global df
    
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        dept_col = 'department' if 'department' in df.columns else 'internship_department'
        if dept_col in df.columns:
            departments = df[dept_col].dropna().unique().tolist()
            return jsonify({"departments": sorted(departments)})
        else:
            return jsonify({"departments": []})
    except Exception as e:
        print(f"Error getting departments: {e}")
        return jsonify({"error": "Error fetching departments"}), 500

@app.route("/locations", methods=["GET"])
def get_locations():
    """Get list of available locations."""
    global df
    
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        if 'location' in df.columns:
            locations = df['location'].dropna().unique().tolist()
            return jsonify({"locations": sorted(locations)})
        else:
            return jsonify({"locations": []})
    except Exception as e:
        print(f"Error getting locations: {e}")
        return jsonify({"error": "Error fetching locations"}), 500

def initialize_app():
    """Initialize the recommender system."""
    global recommender
    if recommender is None:
        if not initialize_recommender():
            print("Warning: Failed to initialize recommender system")

# Initialize before any request if not already initialized
@app.before_request
def ensure_initialized():
    global recommender
    if recommender is None:
        initialize_app()

if __name__ == "__main__":
    # Initialize recommender on startup
    print("Initializing recommender system...")
    success = initialize_recommender()
    if success:
        print("Starting Flask server...")
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("Failed to initialize. Please check that 'final_internship.csv' exists.")