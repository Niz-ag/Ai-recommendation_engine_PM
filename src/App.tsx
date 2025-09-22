import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { ThemeToggle } from './components/ThemeToggle';
import { LandingPage } from './components/LandingPage';
import { RecommendationForm } from './components/RecommendationForm';
import { RecommendationResults } from './components/RecommendationResults';
import { AdminPanel } from './components/AdminPanel';
import { TrendingInternships } from './components/TrendingInternships';
import { UserStats } from './components/UserStats';

// This interface is now corrected
interface UserFormData {
  age: string;
  familyIncome: string;
  skills: string;
  location: string;
  workMode: string;
  paymentPreference?: string;
  topN: number;
}

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

interface ApiStats {
  total_internships: number;
  unique_companies: number;
  unique_locations: number;
  paid_internships: number;
  remote_internships: number;
}

const API_BASE_URL = `${window.location.protocol}//${window.location.hostname}:5000`;

const getUserId = () => {
  let userId = localStorage.getItem('internship_user_id');
  if (!userId) {
    userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('internship_user_id', userId);
  }
  return userId;
};

const LandingWrapper = () => {
  const navigate = useNavigate();
  const [apiStatus, setApiStatus] = useState<'loading' | 'connected' | 'error'>('loading');
  const [stats, setStats] = useState<ApiStats | null>(null);

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/`);
        if (response.ok) {
          setApiStatus('connected');
          const statsResponse = await fetch(`${API_BASE_URL}/stats`);
          if (statsResponse.ok) {
            const statsData = await statsResponse.json();
            setStats(statsData);
          }
        } else {
          setApiStatus('error');
        }
      } catch (error) {
        console.error('API connection failed:', error);
        setApiStatus('error');
      }
    };
    checkApiStatus();
  }, []);

  const handleGetStarted = () => {
    if (apiStatus === 'connected') {
      navigate('/form');
    } else {
      alert('Please wait for the system to initialize or check your connection.');
    }
  };

  return (
    <div className="min-h-screen">
      <LandingPage onGetStarted={handleGetStarted} />
      <div className="fixed bottom-4 right-4 z-50">
        <div
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
            apiStatus === 'loading'
              ? 'bg-yellow-100 text-yellow-800 animate-pulse'
              : apiStatus === 'connected'
              ? 'bg-green-100 text-green-800'
              : 'bg-red-100 text-red-800'
          }`}
        >
          {apiStatus === 'loading' && 'Initializing AI Engine...'}
          {apiStatus === 'connected' && `Ready (${stats?.total_internships || 0} internships)`}
          {apiStatus === 'error' && 'Connection Error'}
        </div>
      </div>
    </div>
  );
};

const FormResultsWrapper = () => {
  const navigate = useNavigate();
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [locations, setLocations] = useState<string[]>([]);
  const [userId] = useState(getUserId());
  const [showUserStats, setShowUserStats] = useState(false);
  const [lastSearchCriteria, setLastSearchCriteria] = useState<UserFormData | null>(null);

  useEffect(() => {
    const loadOptions = async () => {
      try {
        const locResponse = await fetch(`${API_BASE_URL}/locations`);
        if (locResponse.ok) {
          const locData = await locResponse.json();
          setLocations(locData.locations || []);
        }
      } catch (error) {
        console.error('Error loading form options:', error);
      }
    };
    loadOptions();
  }, []);

  const handleFormSubmit = async (formData: UserFormData) => {
    setIsLoading(true);
    setError(null);
    setLastSearchCriteria(formData);

    try {
      const requestData = { ...formData, userId: userId };
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.recommendations && Array.isArray(data.recommendations)) {
        setRecommendations(data.recommendations);
        if (data.recommendations.length === 0) {
          // MODIFIED: More specific error message
          setError('No internships found with at least a 55% match for your skills and location. Please try adjusting your search criteria.');
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError(error instanceof Error ? error.message : 'An unexpected error occurred');
      setRecommendations([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = async (internshipId: string, feedbackType: string, rating?: number) => {
    try {
      await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, internshipId, feedbackType, rating, userProfile: lastSearchCriteria }),
      });
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  return (
    <div className="min-h-screen py-12 px-4 bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <ThemeToggle />

      <div className="max-w-6xl mx-auto mb-8 flex justify-between items-center">
        <button onClick={() => navigate('/')} className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium">
          ← Back to Home
        </button>
        <div className="flex gap-4">
          <button onClick={() => setShowUserStats(!showUserStats)} className="bg-green-500 text-white px-4 py-2 rounded-lg">
            My Stats
          </button>
          <button onClick={() => navigate('/trending')} className="bg-purple-500 text-white px-4 py-2 rounded-lg">
            Trending
          </button>
          <button onClick={() => navigate('/admin')} className="bg-orange-500 text-white px-4 py-2 rounded-lg">
            Add Internship
          </button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto space-y-8">
        {showUserStats && <UserStats userId={userId} onClose={() => setShowUserStats(false)} />}

        <RecommendationForm onSubmit={handleFormSubmit} isLoading={isLoading} locations={locations} />

        {isLoading && <p className="text-center">AI is analyzing internships...</p>}

        {error && <div className="text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400 p-4 rounded-xl text-center font-medium">{error}</div>}

        {!isLoading && !error && recommendations.length > 0 && (
          <RecommendationResults recommendations={recommendations} onFeedback={handleFeedback} />
        )}
      </div>
    </div>
  );
};

const TrendingWrapper = () => {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen py-12 px-4 bg-gradient-to-br from-purple-50 via-pink-50 to-red-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <ThemeToggle />
      <div className="max-w-6xl mx-auto mb-8 flex justify-start items-center">
        <button
          onClick={() => navigate('/form')}
          className="text-purple-600 dark:text-purple-400 hover:text-purple-800 dark:hover:text-purple-300 font-medium"
        >
          ← Back to Search
        </button>
      </div>
      <TrendingInternships />
    </div>
  );
};

const AdminWrapper = () => {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen py-12 px-4 bg-gradient-to-br from-orange-50 via-red-50 to-pink-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <ThemeToggle />
      <div className="max-w-6xl mx-auto mb-8 flex justify-start items-center">
        <button
          onClick={() => navigate('/form')}
          className="text-orange-600 dark:text-orange-400 hover:text-orange-800 dark:hover:text-orange-300 font-medium"
        >
          ← Back to Search
        </button>
      </div>
      <div className="max-w-6xl mx-auto">
        <AdminPanel />
      </div>
    </div>
  );
};

function App() {
  return (
    <ThemeProvider>
      <Router>
        <Routes>
          <Route path="/" element={<LandingWrapper />} />
          <Route path="/form" element={<FormResultsWrapper />} />
          <Route path="/trending" element={<TrendingWrapper />} />
          <Route path="/admin" element={<AdminWrapper />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;