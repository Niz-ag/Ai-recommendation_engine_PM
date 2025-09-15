import React from 'react';
import { MapPin, Building, Clock, Star, ExternalLink, Zap } from 'lucide-react';

interface Recommendation {
  id: string;
  company: string;
  position: string;
  location: string;
  workMode: string;
  duration: string;
  stipend: string;
  matchScore: number;
  description: string;
  requirements: string[];
  benefits: string[];
  logo?: string;
}

interface RecommendationResultsProps {
  recommendations: Recommendation[];
}

export const RecommendationResults: React.FC<RecommendationResultsProps> = ({ recommendations }) => {
  if (recommendations.length === 0) return null;

  return (
    <div className="mt-12">
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full mb-4">
          <Zap className="w-6 h-6 text-white" />
        </div>
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Your Personalized Recommendations
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Found {recommendations.length} perfect matches based on your profile
        </p>
      </div>

      <div className="grid gap-6">
        {recommendations.map((rec, index) => (
          <div
            key={rec.id}
            
            className="bg-white dark:bg-gray-800 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-200 dark:border-gray-700 p-6 transform hover:scale-[1.02]"
          >
            
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center text-white font-bold text-lg">
                  {rec.company.charAt(0)}
                </div>
                <div>
                  <h4 className="text-xl font-bold text-gray-900 dark:text-white">
                    {rec.position}
                  </h4>
                  <p className="text-gray-600 dark:text-gray-400 font-medium">
                    {rec.company}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 bg-green-100 dark:bg-green-900/30 px-3 py-1 rounded-full">
                  <Star className="w-4 h-4 text-green-600 dark:text-green-400 fill-current" />
                  <span className="text-sm font-semibold text-green-700 dark:text-green-400">
                    {rec.matchScore}% Match
                  </span>
                </div>
                {index === 0 && (
                  <div className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white px-3 py-1 rounded-full text-xs font-bold">
                    TOP PICK
                  </div>
                )}
              </div>
            </div>

            <p className="text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">
              {rec.description || "No description available"}
            </p>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="flex items-center gap-2 text-sm">
                <MapPin className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                <span className="text-gray-700 dark:text-gray-300">{rec.location}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Building className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                <span className="text-gray-700 dark:text-gray-300">{rec.workMode || "N/A"}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Clock className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                <span className="text-gray-700 dark:text-gray-300">{rec.duration}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <span className="text-gray-700 dark:text-gray-300">{rec.stipend}</span>
              </div>
            </div>

            {/* Show first 2 requirements + all benefits */}
            <p className="text-gray-600 dark:text-gray-300 mb-6 text-sm">
              Key requirements: {(rec.requirements || []).join(', ')}
              <br />
              Benefits: {(rec.benefits || []).join(', ')}
            </p>

            <div className="flex gap-3">
              <button className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-300 flex items-center justify-center gap-2">
                <ExternalLink className="w-4 h-4" />
                Apply Now
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
