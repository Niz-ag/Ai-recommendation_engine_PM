import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { ThemeToggle } from './components/ThemeToggle';
import { LandingPage } from './components/LandingPage';
import { RecommendationForm } from './components/RecommendationForm';
import { RecommendationResults } from './components/RecommendationResults';

interface UserFormData {
  age: string;
  familyIncome: string;
  skills: string;
  location: string;
  workMode: string;
  duration: string;
}

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
}

interface ApiStats {
  total_internships: number;
  unique_companies: number;
  unique_locations: number;
  work_modes: Record<string, number>;
  paid_vs_unpaid: {
    paid: number;
    unpaid: number;
  };
}

const API_BASE_URL = `${window.location.protocol}//${window.location.hostname}:5000`;

// Landing Page wrapper
const LandingWrapper = () => {
  const navigate = useNavigate();
  const [apiStatus, setApiStatus] = useState<'loading' | 'connected' | 'error'>('loading');
  const [stats, setStats] = useState<ApiStats | null>(null);

  useEffect(() => {
    // Check API status and get stats
    const checkApiStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/`);
        if (response.ok) {
          setApiStatus('connected');
          // Get stats
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
      
      {/* API Status Indicator */}
      <div className="fixed bottom-4 right-4 z-50">
        <div className={`px-4 py-2 rounded-lg text-sm font-medium ${
          apiStatus === 'loading' ? 'bg-yellow-100 text-yellow-800' :
          apiStatus === 'connected' ? 'bg-green-100 text-green-800' :
          'bg-red-100 text-red-800'
        }`}>
          {apiStatus === 'loading' && 'Initializing...'}
          {apiStatus === 'connected' && `Ready (${stats?.total_internships || 0} internships)`}
          {apiStatus === 'error' && 'Connection Error'}
        </div>
      </div>
    </div>
  );
};

// Form + Results wrapper
const FormResultsWrapper = () => {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [locations, setLocations] = useState<string[]>([]);

  // Load departments and locations on component mount
  useEffect(() => {
    const loadOptions = async () => {
      try {
        // Load locations
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

    try {
      console.log('Sending form data:', formData);
      
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Received recommendations:', data);
      
      if (data.recommendations && Array.isArray(data.recommendations)) {
        setRecommendations(data.recommendations);
        if (data.recommendations.length === 0) {
          setError('No matching internships found. Try adjusting your criteria.');
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setError(error instanceof Error ? error.message : 'An unexpected error occurred');
      setRecommendations([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    setError(null);
    setRecommendations([]);
  };

  return (
    <div className="min-h-screen py-12 px-4 bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <ThemeToggle />
      <div className="max-w-4xl mx-auto space-y-8">
        <RecommendationForm 
          onSubmit={handleFormSubmit} 
          isLoading={isLoading}
          locations={locations}
        />
        
        {isLoading && (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600 dark:text-gray-400">
              Finding the best internships for you...
            </p>
          </div>
        )}
        
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800 dark:text-red-400">
                  Error occurred
                </h3>
                <div className="mt-2 text-sm text-red-700 dark:text-red-300">
                  {error}
                </div>
                <div className="mt-4">
                  <button
                    onClick={handleRetry}
                    className="bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-400 px-3 py-1 rounded text-sm hover:bg-red-200 dark:hover:bg-red-900/60 transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {!isLoading && !error && recommendations.length > 0 && (
          <>
            <div className="text-center">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                Found {recommendations.length} Great Matches!
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Here are the internships that best match your preferences
              </p>
            </div>
            <RecommendationResults recommendations={recommendations} />
          </>
        )}
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
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;