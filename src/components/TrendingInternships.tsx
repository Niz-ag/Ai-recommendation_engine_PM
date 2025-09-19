import React, { useState, useEffect } from 'react';
import { TrendingUp, Users, Eye, Clock } from 'lucide-react';

interface TrendingInternship {
  id: string;
  title: string;
  company: string;
  location: string;
  workMode: string;
  duration: string;
  stipend: string;
  description: string;
  recent_interactions: number;
  recent_upvotes: number;
  recent_applications: number;
  requirements: string[];
  benefits: string[];
}

const API_BASE_URL = `${window.location.protocol}//${window.location.hostname}:5000`;

export const TrendingInternships: React.FC = () => {
  const [trending, setTrending] = useState<TrendingInternship[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timePeriod, setTimePeriod] = useState(7);

  useEffect(() => {
    const fetchTrending = async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `${API_BASE_URL}/trending?days=${timePeriod}&limit=15`
        );
        if (response.ok) {
          const data = await response.json();
          setTrending(data.trending || []);
        } else {
          setError('Failed to fetch trending internships');
        }
      } catch (err) {
        setError('Error loading trending internships');
        console.error('Error fetching trending internships:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchTrending();
  }, [timePeriod]);

  const getTrendingLevel = (interactions: number) => {
    if (interactions >= 20)
      return { level: 'Hot', style: 'red' };
    if (interactions >= 10)
      return { level: 'Trending', icon: TrendingUp, style: 'orange' };
    return { level: 'Rising', icon: Eye, style: 'blue' };
  };

  const badgeStyles: Record<
    string,
    { bg: string; text: string; icon: string }
  > = {
    red: {
      bg: 'bg-red-100 dark:bg-red-900/30',
      text: 'text-red-700 dark:text-red-400',
      icon: 'text-red-600 dark:text-red-400'
    },
    orange: {
      bg: 'bg-orange-100 dark:bg-orange-900/30',
      text: 'text-orange-700 dark:text-orange-400',
      icon: 'text-orange-600 dark:text-orange-400'
    },
    blue: {
      bg: 'bg-blue-100 dark:bg-blue-900/30',
      text: 'text-blue-700 dark:text-blue-400',
      icon: 'text-blue-600 dark:text-blue-400'
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-purple-600 border-t-transparent mb-4"></div>
        <p className="text-gray-600 dark:text-gray-400 text-lg">
          Loading trending internships...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
          ERROR</div>
        <p className="text-red-600 dark:text-red-400 text-lg">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full mb-4">
          <TrendingUp className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Trending Internships
        </h1>
        <p className="text-gray-600 dark:text-gray-400 text-lg">
          Popular opportunities based on community engagement
        </p>
      </div>

      {/* Time Period Selector */}
      <div className="flex justify-center">
        <div className="bg-white dark:bg-gray-800 p-1 rounded-xl shadow-lg">
          {[3, 7, 14, 30].map((days) => (
            <button
              key={days}
              onClick={() => setTimePeriod(days)}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                timePeriod === days
                  ? 'bg-purple-600 text-white shadow-md'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-purple-50 dark:hover:bg-gray-700'
              }`}
            >
              Last {days} days
            </button>
          ))}
        </div>
      </div>

      {/* Trending List */}
      {trending.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
            <Clock className="w-8 h-8 text-gray-400" />
          </div>
          <p className="text-gray-500 dark:text-gray-400 text-lg">
            No trending internships in the selected period
          </p>
        </div>
      ) : (
        <div className="grid gap-6">
          {trending.map((internship, index) => {
            const trendingData = getTrendingLevel(
              internship.recent_interactions
            );
            const TrendIcon = trendingData.icon;
            const style = badgeStyles[trendingData.style];

            return (
              <div
                key={internship.id}
                className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-200 dark:border-gray-700 p-6"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center text-white font-bold text-lg">
                      {internship.company.charAt(0).toUpperCase()}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                        {internship.title}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 font-medium">
                        {internship.company}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        #{index + 1}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        Rank
                      </div>
                    </div>
                    <div
                      className={`flex items-center gap-1 px-3 py-1 rounded-full ${style.bg}`}
                    >
                      <TrendIcon className={`w-4 h-4 ${style.icon}`} />
                      <span
                        className={`text-sm font-semibold ${style.text}`}
                      >
                        {trendingData.level}
                      </span>
                    </div>
                  </div>
                </div>

                <p className="text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">
                  {internship.description}
                </p>

                {/* Stats Grid */}
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <Users className="w-5 h-5 text-blue-500 mx-auto mb-1" />
                    <div className="text-lg font-bold text-blue-600 dark:text-blue-400">
                      {internship.recent_interactions}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Interactions
                    </div>
                  </div>
                  <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <TrendingUp className="w-5 h-5 text-green-500 mx-auto mb-1" />
                    <div className="text-lg font-bold text-green-600 dark:text-green-400">
                      {internship.recent_upvotes}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Upvotes
                    </div>
                  </div>
                  <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                    UP
                    <div className="text-lg font-bold text-purple-600 dark:text-purple-400">
                      {internship.recent_applications}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Applications
                    </div>
                  </div>
                </div>

                {/* Basic Details */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-sm">
                    <span className="text-gray-500 dark:text-gray-400">
                      Location:
                    </span>
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {internship.location}
                    </div>
                  </div>
                  <div className="text-sm">
                    <span className="text-gray-500 dark:text-gray-400">
                      Mode:
                    </span>
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {internship.workMode}
                    </div>
                  </div>
                  <div className="text-sm">
                    <span className="text-gray-500 dark:text-gray-400">
                      Duration:
                    </span>
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {internship.duration}
                    </div>
                  </div>
                  <div className="text-sm">
                    <span className="text-gray-500 dark:text-gray-400">
                      Stipend:
                    </span>
                    <div className="font-semibold text-gray-900 dark:text-white">
                      {internship.stipend}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
