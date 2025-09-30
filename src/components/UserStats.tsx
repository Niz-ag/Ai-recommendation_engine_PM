import React, { useState, useEffect } from 'react';
import { X, ThumbsUp, Send, TrendingUp, Award } from 'lucide-react';

interface UserStatsProps {
  userId: string;
  onClose: () => void;
}

interface UserStats {
  total_interactions: number;
  upvotes: number;
  downvotes: number;
  applications: number;
  avg_rating: number;
}

const API_BASE_URL = `${window.location.protocol}//${window.location.hostname}:5000`;

export const UserStats: React.FC<UserStatsProps> = ({ userId, onClose }) => {
  const [stats, setStats] = useState<UserStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/user/stats/${userId}`);
        if (response.ok) {
          const data = await response.json();
          setStats(data);
        } else {
          setError('Failed to fetch user statistics');
        }
      } catch (err) {
        setError('Error loading statistics');
        console.error('Error fetching user stats:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, [userId]);

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 max-w-md w-full mx-4">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent mx-auto mb-4"></div>
            <p className="text-gray-600 dark:text-gray-400">Loading your stats...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 max-w-md w-full mx-4">
          <div className="text-center">
            <div className="w-12 h-12 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
              <X className="w-6 h-6 text-red-600 dark:text-red-400" />
            </div>
            <p className="text-red-600 dark:text-red-400 mb-4">{error}</p>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  const getEngagementLevel = () => {
    if (!stats || stats.total_interactions === 0) return { level: 'New User', color: 'gray', description: 'Start exploring internships!' };
    if (stats.total_interactions < 5) return { level: 'Explorer', color: 'blue', description: 'Getting started' };
    if (stats.total_interactions < 15) return { level: 'Active User', color: 'green', description: 'Regularly engaged' };
    if (stats.total_interactions < 30) return { level: 'Power User', color: 'purple', description: 'Highly engaged' };
    return { level: 'Super User', color: 'yellow', description: 'Top contributor' };
  };

  const engagement = getEngagementLevel();

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Your Activity Stats</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="w-6 h-6 text-gray-500 dark:text-gray-400" />
          </button>
        </div>

        {/* User Level Badge */}
        <div className="text-center mb-8">
          <div className={`inline-flex items-center gap-2 px-4 py-2 bg-${engagement.color}-100 dark:bg-${engagement.color}-900/30 rounded-full mb-2`}>
            <Award className={`w-5 h-5 text-${engagement.color}-600 dark:text-${engagement.color}-400`} />
            <span className={`font-semibold text-${engagement.color}-700 dark:text-${engagement.color}-400`}>
              {engagement.level}
            </span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">{engagement.description}</p>
        </div>

        {stats && (
          <div className="space-y-6">
            {/* Main Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl text-center">
                <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-3">
                  <TrendingUp className="w-6 h-6 text-white" />
                </div>
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {stats.total_interactions}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Total Interactions</div>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl text-center">
                <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-3">
                  <ThumbsUp className="w-6 h-6 text-white" />
                </div>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {stats.upvotes}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Upvotes</div>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl text-center">
                <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Send className="w-6 h-6 text-white" />
                </div>
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {stats.applications}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Applications</div>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl text-center">
                <div className="w-12 h-12 bg-orange-500 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Award className="w-6 h-6 text-white" />
                </div>
                <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                  {stats.avg_rating ? stats.avg_rating.toFixed(1) : 'N/A'}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Avg Rating</div>
              </div>
            </div>

            {/* Insights */}
            {stats.total_interactions > 0 && (
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-700 dark:to-gray-600 p-6 rounded-xl">
                <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Your Impact</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-300">Positive Feedback Rate</span>
                    <span className="font-semibold text-green-600 dark:text-green-400">
                      {stats.total_interactions > 0 ? Math.round((stats.upvotes / stats.total_interactions) * 100) : 0}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 dark:text-gray-300">Application Rate</span>
                    <span className="font-semibold text-blue-600 dark:text-blue-400">
                      {stats.total_interactions > 0 ? Math.round((stats.applications / stats.total_interactions) * 100) : 0}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700/50 rounded-xl">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {stats.total_interactions === 0 
                  ? "Start interacting with internships to see your personalized stats!"
                  : "Your feedback helps improve recommendations for everyone. Keep exploring!"
                }
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};