import React, { useState } from 'react';
import { MapPin, Building, Clock, Star, ThumbsUp, ThumbsDown, ExternalLink, Zap, Brain, Award } from 'lucide-react';

interface Recommendation {
  id: string;
  title: string;
  company: string;
  location: string;
  workMode: string;
  duration: string;
  stipend: string;
  matchScore: number;
  description: string;
  requirements: string[];
  benefits: string[];
  skillsMatch: number;
  locationMatch: number;
  collaborativeScore: number;
  isPaid: boolean;
}

interface RecommendationResultsProps {
  recommendations: Recommendation[];
  onFeedback: (internshipId: string, feedbackType: string, rating?: number) => void;
}

export const RecommendationResults: React.FC<RecommendationResultsProps> = ({ 
  recommendations, 
  onFeedback 
}) => {
  const [feedbackGiven, setFeedbackGiven] = useState<Record<string, string>>({});

  if (recommendations.length === 0) return null;

  const handleFeedback = (id: string, type: string) => {
    onFeedback(id, type);
    setFeedbackGiven(prev => ({ ...prev, [id]: type }));
  };


  return (
    <div className="space-y-6">
      {recommendations.map((rec, index) => (
        <div
          key={rec.id}
          className="group bg-white dark:bg-gray-800 rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-500 border border-gray-200 dark:border-gray-700 p-8 transform hover:scale-[1.02] overflow-hidden relative"
        >
          {/* Gradient Background Accent */}
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-purple-500"></div>
          
          {/* Header Section */}
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center text-white font-bold text-xl shadow-lg">
                {rec.company.charAt(0).toUpperCase()}
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
                  {rec.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 font-semibold text-lg">
                  {rec.company}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 px-4 py-2 rounded-full">
                <Star className="w-5 h-5 text-green-600 dark:text-green-400 fill-current" />
                <span className="text-lg font-bold text-green-700 dark:text-green-400">
                  {rec.matchScore}% Match
                </span>
              </div>
              {index === 0 && (
                <div className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white px-3 py-1 rounded-full text-sm font-bold shadow-lg animate-pulse">
                  TOP PICK
                </div>
              )}
            </div>
          </div>

          {/* Description */}
          <p className="text-gray-600 dark:text-gray-300 mb-6 leading-relaxed text-lg">
            {rec.description}
          </p>

          {/* Key Details Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-6">
            <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
              <MapPin className="w-5 h-5 text-blue-500" />
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Location</p>
                <p className="text-sm font-semibold text-gray-900 dark:text-white">{rec.location}</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
              <Building className="w-5 h-5 text-purple-500" />
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Work Mode</p>
                <p className="text-sm font-semibold text-gray-900 dark:text-white">{rec.workMode}</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
              <Clock className="w-5 h-5 text-green-500" />
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Duration</p>
                <p className="text-sm font-semibold text-gray-900 dark:text-white">{rec.duration}</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
              <Award className="w-5 h-5 text-orange-500" />
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Stipend</p>
                <p className="text-sm font-semibold text-gray-900 dark:text-white">{rec.stipend}</p>
              </div>
            </div>
          </div>

          

          {/* Requirements and Benefits */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <Zap className="w-4 h-4 text-orange-500" />
                Key Requirements
              </h4>
              <div className="flex flex-wrap gap-2">
                {rec.requirements.map((req, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-full text-sm font-medium"
                  >
                    {req}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <Award className="w-4 h-4 text-green-500" />
                Benefits
              </h4>
              <div className="space-y-2">
                {rec.benefits.slice(0, 3).map((benefit, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-gray-600 dark:text-gray-300">{benefit}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex gap-3">
              <button
                onClick={() => handleFeedback(rec.id, 'upvote')}
                disabled={feedbackGiven[rec.id] === 'upvote'}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  feedbackGiven[rec.id] === 'upvote'
                    ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-green-100 dark:hover:bg-green-900/30'
                }`}
              >
                <ThumbsUp className="w-4 h-4" />
                {feedbackGiven[rec.id] === 'upvote' ? 'Liked' : 'Like'}
              </button>
              <button
                onClick={() => handleFeedback(rec.id, 'downvote')}
                disabled={feedbackGiven[rec.id] === 'downvote'}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  feedbackGiven[rec.id] === 'downvote'
                    ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-red-100 dark:hover:bg-red-900/30'
                }`}
              >
                <ThumbsDown className="w-4 h-4" />
                {feedbackGiven[rec.id] === 'downvote' ? 'Disliked' : 'Dislike'}
              </button>
            </div>
            
            <div className="flex gap-3">
              <button 
                onClick={() => handleFeedback(rec.id, 'apply')}
                className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                <ExternalLink className="w-5 h-5" />
                Apply Now
              </button>
            </div>
          </div>
        </div>
      ))}
      
      {/* Feedback Reminder */}
      <div className="text-center p-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-800 dark:to-gray-700 rounded-2xl">
        <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400 mx-auto mb-3" />
        <p className="text-gray-600 dark:text-gray-300 font-medium">
          Your feedback helps our AI learn and provide better recommendations for everyone!
        </p>
      </div>
    </div>
  );
};