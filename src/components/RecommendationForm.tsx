import React, { useState } from 'react';

interface UserFormData {
  age: string;
  familyIncome: string;
  skills: string;
  location: string;
  // department: string;
  workMode: string;
  duration: string;
  gender?: string;
}

interface RecommendationFormProps {
  onSubmit: (data: UserFormData) => void;
  isLoading: boolean;
  // departments?: string[];
  locations?: string[];
}

export const RecommendationForm: React.FC<RecommendationFormProps> = ({
  onSubmit,
  isLoading,
  // departments = [],
  locations = []
}) => {
  const [formData, setFormData] = useState<UserFormData>({
    age: '',
    familyIncome: '',
    skills: '',
    location: '',
    // department: '',
    workMode: 'remote',
    duration: '',
    gender: 'any'
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Form submission data:', formData);
    onSubmit(formData);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
        Find Your Perfect Internship
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
       
        <div>
          <label htmlFor="skills" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Skills *
          </label>
          <textarea
            id="skills"
            name="skills"
            value={formData.skills}
            onChange={handleChange}
            rows={3}
            placeholder="e.g., Python, JavaScript, React, Marketing, Communication"
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          />
        </div>

        {/* Location Field */}
        <div>
          <label htmlFor="location" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Preferred Location
          </label>
          {locations.length > 0 ? (
            <select
              id="location"
              name="location"
              value={formData.location}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="">Any location</option>
              {locations.map((loc) => (
                <option key={loc} value={loc}>{loc}</option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              id="location"
              name="location"
              value={formData.location}
              onChange={handleChange}
              placeholder="e.g., New York, London, Remote"
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            />
          )}
        </div>

        {/* Work Mode */}
        <div>
          <label htmlFor="workMode" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Work Mode
          </label>
          <select
            id="workMode"
            name="workMode"
            value={formData.workMode}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            <option value="remote">Remote</option>
            <option value="onsite">Onsite</option>
            <option value="hybrid">Hybrid</option>
          </select>
        </div>

        {/* Age */}
        <div>
          <label htmlFor="age" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Age
          </label>
          <input
            type="number"
            id="age"
            name="age"
            value={formData.age}
            onChange={handleChange}
            min="21"
            max="24"
            placeholder="e.g., 22"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          />
        </div>

        {/* Duration */}
        <div>
          <label htmlFor="duration" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Preferred Duration
          </label>
          <select
            id="duration"
            name="duration"
            value={formData.duration}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            <option value="">Any duration</option>
            <option value="1 month">1 Month</option>
            <option value="2 months">2 Months</option>
            <option value="3 months">3 Months</option>
            <option value="6 months">6 Months</option>
            <option value="12 months">12 Months</option>
          </select>
        </div>

        {/* Family Income */}
        <div>
          <label htmlFor="familyIncome" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Family Income Range
          </label>
          <select
            id="familyIncome"
            name="familyIncome"
            value={formData.familyIncome}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            <option value="">Prefer not to say</option>
            <option value="0-25000">₹0 - ₹1 lac</option>
            <option value="25000-50000">₹1 lac - ₹3 lacs</option>
            <option value="50000-75000">₹3 lacs - ₹6 lacs</option>
            <option value="75000-100000">₹6 lacs - ₹8lacs</option>
            <option value="100000+">₹8 lacs+</option>
          </select>
        </div>

        {/* Gender */}
        <div>
          <label htmlFor="gender" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Gender Preference
          </label>
          <select
            id="gender"
            name="gender"
            value={formData.gender}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            <option value="any">Any</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium py-2 px-4 rounded-md transition duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          {isLoading ? 'Finding Internships...' : 'Find Internships'}
        </button>
      </form>
    </div>
  );
};