from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import os
import sys
import traceback
from datetime import datetime
import json

# Import the enhanced engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine2 import OptimizedInternshipRecommender, Config

app = Flask(__name__)
CORS(app)

# Global variables
df = None
recommender = None

def initialize_recommender():
    global df, recommender
    try:
        print("Loading CSV file...")
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_internship.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_internship.csv")

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} internships")
        print(f"Columns: {list(df.columns)}")

        print("Initializing enhanced recommender with collaborative filtering...")
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
    """Enhanced formatting with more details."""
    # Extract basic information
    title = row.get('internship_title') or row.get('title', 'Unknown Title')
    company_val = row.get("company_name") or row.get("company", "Unknown Company")
    stipend_val = row.get("stipend", "N/A")
    duration_val = row.get("duration", "3 Months")
    skills_val = row.get("required_skills") or row.get("skills") or row.get("Skills", "")
    description_val = row.get("description", "Exciting internship opportunity to gain hands-on experience")
    location_val = row.get("location", "Not specified")
    work_mode_val = row.get("work_mode", "Remote")
    
    # Determine work mode from location if not explicitly set
    if pd.isna(work_mode_val) or work_mode_val == "Remote":
        if any(keyword in str(location_val).lower() for keyword in ['remote', 'work from home', 'wfh']):
            work_mode_val = "Remote"
        else:
            work_mode_val = "Onsite"

    # Process requirements
    requirements = []
    if skills_val and not pd.isna(skills_val):
        skill_list = [skill.strip() for skill in str(skills_val).split(",") if skill.strip()]
        requirements = skill_list[:4]  # Take first 4 skills
    
    if not requirements:
        requirements = ["Basic computer skills", "Communication skills", "Enthusiasm to learn"]

    # Process benefits
    benefits = ["Professional development", "Mentorship program", "Certificate of completion"]
    if row.get("is_paid", True):
        benefits.insert(0, f"Stipend: {stipend_val}")
    else:
        benefits.insert(0, "Valuable work experience")

    return {
        "id": str(row.get('internship_id', index)),
        "title": title,
        "company": company_val,
        "location": location_val,
        "workMode": work_mode_val,
        "duration": duration_val,
        "stipend": stipend_val,
        "matchScore": round(row.get("match_score", 0) * 100, 1),
        "description": description_val,
        "requirements": requirements,
        "benefits": benefits,
        "skillsMatch": round(row.get("skills_exact_similarity", 0) * 100, 1),
        "locationMatch": round(row.get("location_similarity", 0) * 100, 1),
        "collaborativeScore": round(row.get("collaborative_score", 0) * 100, 1),
        "isPaid": row.get("is_paid", False)
    }

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Enhanced Internship Recommendation API with Collaborative Filtering",
        "status": "active",
        "recommender_loaded": recommender is not None,
        "features": [
            "Collaborative Filtering",
            "Skills Matching",
            "Location Preferences", 
            "User Feedback Learning",
            "Trending Internships"
        ]
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    global recommender
    if recommender is None:
        return jsonify({"error": "Recommendation engine not initialized", "recommendations": []}), 500

    try:
        data = request.get_json()
        print("Received request:", data)

        # Extract parameters
        location = data.get("location", "")
        skills = data.get("skills", "")
        gender = data.get("gender", "any")
        work_mode = data.get("workMode", "remote")
        payment_preference = data.get("paymentPreference", "any")
        user_id = data.get("userId", None)
        
        print(f"Parameters: location={location}, skills={skills}, gender={gender}, work_mode={work_mode}")

        # Get recommendations using enhanced engine
        recommendations_df = recommender.recommend_internships(
            user_location=location,
            user_skills=skills,
            user_gender=gender,
            user_payment_preference=payment_preference,
            user_id=user_id,
            top_n=20
        )

        recommendations = [
            format_recommendation(row, idx + 1) 
            for idx, (_, row) in enumerate(recommendations_df.iterrows())
        ]
        
        print(f"Returning {len(recommendations)} recommendations")
        return jsonify({
            "recommendations": recommendations,
            "total_found": len(recommendations),
            "search_criteria": {
                "skills": skills,
                "location": location,
                "work_mode": work_mode,
                "gender": gender
            }
        })

    except Exception as e:
        print(f"Error in recommend endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "recommendations": []}), 500

@app.route("/feedback", methods=["POST"])
def add_feedback():
    """Add user feedback for collaborative filtering."""
    global recommender
    if recommender is None:
        return jsonify({"error": "Recommendation engine not initialized"}), 500

    try:
        data = request.get_json()
        user_id = data.get("userId")
        internship_id = data.get("internshipId")
        feedback_type = data.get("feedbackType")  # 'upvote', 'downvote', 'apply', 'skip'
        rating = data.get("rating")  # Optional 1-5 rating
        user_profile = data.get("userProfile")  # Optional user profile data

        if not all([user_id, internship_id, feedback_type]):
            return jsonify({"error": "Missing required fields"}), 400

        # Add feedback to the system
        recommender.add_user_feedback(
            user_id=user_id,
            internship_id=internship_id,
            feedback_type=feedback_type,
            user_profile=user_profile,
            rating=rating
        )

        return jsonify({
            "message": "Feedback added successfully",
            "status": "success"
        })

    except Exception as e:
        print(f"Error adding feedback: {e}")
        return jsonify({"error": "Failed to add feedback"}), 500

@app.route("/user/stats/<user_id>", methods=["GET"])
def get_user_stats(user_id):
    """Get user interaction statistics."""
    global recommender
    if recommender is None:
        return jsonify({"error": "Recommendation engine not initialized"}), 500

    try:
        stats = recommender.get_user_statistics(user_id)
        return jsonify(stats)
    except Exception as e:
        print(f"Error getting user stats: {e}")
        return jsonify({"error": "Failed to get user statistics"}), 500

@app.route("/trending", methods=["GET"])
def get_trending():
    """Get trending internships."""
    global recommender
    if recommender is None:
        return jsonify({"error": "Recommendation engine not initialized"}), 500

    try:
        days = request.args.get('days', 7, type=int)
        top_n = request.args.get('limit', 10, type=int)
        
        trending_df = recommender.get_trending_internships(days=days, top_n=top_n)
        
        trending = [
            {
                **format_recommendation(row, idx + 1),
                "recent_interactions": row.get("recent_interactions", 0),
                "recent_upvotes": row.get("recent_upvotes", 0),
                "recent_applications": row.get("recent_applications", 0)
            }
            for idx, (_, row) in enumerate(trending_df.iterrows())
        ]
        
        return jsonify({
            "trending": trending,
            "period_days": days
        })
        
    except Exception as e:
        print(f"Error getting trending internships: {e}")
        return jsonify({"error": "Failed to get trending internships"}), 500

@app.route("/internships", methods=["POST"])
def add_internship():
    """Add a new internship to the system."""
    global df, recommender
    
    try:
        data = request.get_json()
        
        # Required fields validation
        required_fields = ['title', 'company', 'skills', 'location', 'duration']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Create new internship record
        new_internship = {
            'internship_id': f"new_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'internship_title': data.get('title'),
            'company_name': data.get('company'),
            'location': data.get('location'),
            'required_skills': data.get('skills'),
            'duration': data.get('duration'),
            'stipend': data.get('stipend', 'Not specified'),
            'description': data.get('description', ''),
            'gender': data.get('gender', 'any'),
            'created_at': datetime.now().isoformat()
        }
        
        # Add to dataframe
        new_df = pd.concat([df, pd.DataFrame([new_internship])], ignore_index=True)
        
        # Save to CSV
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "final_internship.csv")
        new_df.to_csv(csv_path, index=False)
        
        # Update global dataframe
        df = new_df
        
        # Reinitialize recommender with new data
        recommender = OptimizedInternshipRecommender(df)
        
        return jsonify({
            "message": "Internship added successfully",
            "internship_id": new_internship['internship_id'],
            "total_internships": len(df)
        })
        
    except Exception as e:
        print(f"Error adding internship: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to add internship"}), 500

@app.route("/admin", methods=["GET"])
def admin_panel():
    """Simple admin panel for adding internships."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Internship Admin Panel</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            textarea { height: 100px; resize: vertical; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .success { color: green; padding: 10px; background: #d4edda; border-radius: 4px; margin: 10px 0; }
            .error { color: red; padding: 10px; background: #f8d7da; border-radius: 4px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Add New Internship</h1>
            <form id="internshipForm">
                <div class="form-group">
                    <label for="title">Title *</label>
                    <input type="text" id="title" name="title" required>
                </div>
                <div class="form-group">
                    <label for="company">Company *</label>
                    <input type="text" id="company" name="company" required>
                </div>
                <div class="form-group">
                    <label for="location">Location *</label>
                    <input type="text" id="location" name="location" required>
                </div>
                <div class="form-group">
                    <label for="skills">Required Skills * (comma separated)</label>
                    <textarea id="skills" name="skills" placeholder="Python, JavaScript, React, etc." required></textarea>
                </div>
                <div class="form-group">
                    <label for="duration">Duration *</label>
                    <select id="duration" name="duration" required>
                        <option value="">Select Duration</option>
                        <option value="1 Month">1 Month</option>
                        <option value="2 Months">2 Months</option>
                        <option value="3 Months">3 Months</option>
                        <option value="6 Months">6 Months</option>
                        <option value="12 Months">12 Months</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="stipend">Stipend</label>
                    <input type="text" id="stipend" name="stipend" placeholder="e.g., ₹15000/month">
                </div>
                <div class="form-group">
                    <label for="description">Description</label>
                    <textarea id="description" name="description" placeholder="Brief description of the internship..."></textarea>
                </div>
                <div class="form-group">
                    <label for="gender">Gender Preference</label>
                    <select id="gender" name="gender">
                        <option value="any">Any</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                    </select>
                </div>
                <button type="submit">Add Internship</button>
            </form>
            <div id="message"></div>
        </div>
        
        <script>
            document.getElementById('internshipForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());
                
                try {
                    const response = await fetch('/internships', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    const messageDiv = document.getElementById('message');
                    
                    if (response.ok) {
                        messageDiv.innerHTML = `<div class="success">${result.message}</div>`;
                        e.target.reset();
                    } else {
                        messageDiv.innerHTML = `<div class="error">${result.error}</div>`;
                    }
                } catch (error) {
                    document.getElementById('message').innerHTML = 
                        `<div class="error">Error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

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
            "paid_internships": len(df[df.get("is_paid", False)]) if "is_paid" in df.columns else 0,
            "remote_internships": len(df[df["location"].str.contains("remote|work from home", case=False, na=False)]) if "location" in df.columns else 0,
        }
        return jsonify(stats)
    except Exception as e:
        print(f"Error generating stats: {e}")
        return jsonify({"error": "Error generating statistics"}), 500

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
    print("Initializing enhanced recommender system...")
    if initialize_recommender():
        print("Starting Flask server with collaborative filtering...")
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("Failed to initialize recommender. Ensure 'final_internship.csv' exists.")