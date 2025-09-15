from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import sys
import traceback

# Add the parent directory to the Python path to import engine.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import OptimizedInternshipRecommender

app = Flask(__name__)
CORS(app)

# Global variables
df = None
recommender = None


def initialize_recommender():
    global df, recommender
    try:
        print("Loading CSV file...")
        # Look for CSV in project root
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_internship.csv")
        if not os.path.exists(csv_path):
            # Fallback to current directory
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_internship.csv")

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} internships")
        print(f"Columns: {list(df.columns)}")

        print("Initializing recommender...")
        recommender = OptimizedInternshipRecommender(df)
        print("Recommender initialized successfully")
        return True
    except FileNotFoundError:
        print("Error: final_internship.csv not found")
        return False
    except Exception as e:
        print(f"Error initializing recommender: {e}")
        traceback.print_exc()
        return False


def format_recommendation(row, index):
    company_val = row.get("company_name") or row.get("company", "Unknown Company")
    stipend_val = row.get("stipend", "N/A")
    duration_val = row.get("duration", "3 Months")
    work_mode_val = row.get("work_mode", "Remote")
    skills_val = row.get("required_skills") or row.get("skills") or row.get("Skills", "")
    description_val = row.get("description", "Exciting internship opportunity")
    location_val = row.get("location", "Not specified")


    # Requirements
    requirements = [skill.strip() for skill in str(skills_val).split(",")[:3]] if skills_val else ["Basic computer skills", "Communication skills"]

    # Benefits
    benefits = ["Professional development", "Mentorship program", "Certificate of completion"]
    if row.get("is_paid", True):
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
        "matchScore": round(row.get("match_score", 0) * 100, 1),
        "description": description_val,
        "requirements": requirements,
        "benefits": benefits,
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Internship Recommendation API is running!",
        "status": "active",
        "recommender_loaded": recommender is not None
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    global recommender
    if recommender is None:
        return jsonify({"error": "Recommendation engine not initialized", "recommendations": []}), 500

    try:
        data = request.get_json()
        print("Received request:", data)

        location = data.get("location", "")
        skills = data.get("skills", "")
        gender = data.get("gender", "any")
        work_mode = data.get("workMode", "remote")
        print([location,skills,gender,work_mode])

        recommendations_df = recommender.recommend_internships(
            user_location=location,
            # user_work_mode=work_mode,
            user_skills=skills,
            user_gender=gender,
            user_payment_preference="any",
            user_mode= work_mode,
            # top_n=10,
            # min_score=0.0
        )

        recommendations = [format_recommendation(row, idx + 1) for idx, (_, row) in enumerate(recommendations_df.iterrows())]
        print(f"Returning {len(recommendations)} recommendations")
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        print(f"Error in recommend endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "recommendations": []}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    global df
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    try:
        stats = {
            "total_internships": len(df),
            "unique_companies": df["company_name"].nunique() if "company_name" in df.columns else 0,
            "unique_locations": df["location"].nunique() if "location" in df.columns else 0,
            "work_modes": df["work_mode"].value_counts().to_dict() if "work_mode" in df.columns else {},
            "paid_vs_unpaid": {
                "paid": len(df[df["is_paid"] == True]) if "is_paid" in df.columns else 0,
                "unpaid": len(df[df["is_paid"] == False]) if "is_paid" in df.columns else 0,
            },
        }
        return jsonify(stats)
    except Exception as e:
        print(f"Error generating stats: {e}")
        return jsonify({"error": "Error generating statistics"}), 500


@app.route("/departments", methods=["GET"])
def get_departments():
    global df
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    try:
        dept_col = "department" if "department" in df.columns else "internship_department"
        departments = df[dept_col].dropna().unique().tolist() if dept_col in df.columns else []
        return jsonify({"departments": sorted(departments)})
    except Exception as e:
        print(f"Error getting departments: {e}")
        return jsonify({"error": "Error fetching departments"}), 500


@app.route("/locations", methods=["GET"])
def get_locations():
    global df
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    try:
        locations = df["location"].dropna().unique().tolist() if "location" in df.columns else []
        return jsonify({"locations": sorted(locations)})
    except Exception as e:
        print(f"Error getting locations: {e}")
        return jsonify({"error": "Error fetching locations"}), 500


@app.before_request
def ensure_initialized():
    global recommender
    if recommender is None:
        initialize_recommender()


if __name__ == "__main__":
    print("Initializing recommender system...")
    if initialize_recommender():
        print("Starting Flask server...")
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("Failed to initialize recommender. Ensure 'final_internship.csv' exists.")
