import React, { useState, useCallback, useMemo } from 'react';
import { User, MapPin, Briefcase, DollarSign } from 'lucide-react';

interface UserFormData {
  age: string;
  familyIncome: string;
  skills: string;
  location: string;
  workMode: string;
  duration: string;
  gender: string;
  paymentPreference: string;
}

interface RecommendationFormProps {
  onSubmit: (data: UserFormData) => void;
  isLoading: boolean;
  locations?: string[];
}

export const RecommendationForm: React.FC<RecommendationFormProps> = React.memo(({
  onSubmit,
  isLoading,
  locations = []
}) => {
  const [formData, setFormData] = useState<UserFormData>({
    age: '',
    familyIncome: '',
    skills: '',
    location: '',
    workMode: 'remote',
    duration: '',
    gender: 'any',
    paymentPreference: 'any'
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  // Memoize the change handler to prevent re-renders
  const handleChange = useCallback((
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

    // Clear error for this field if it exists
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  }, [errors]);

  const validateForm = useCallback((): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.skills.trim()) {
      newErrors.skills = 'Please enter your skills';
    }

    if (formData.workMode !== 'remote' && !formData.location.trim()) {
      newErrors.location = 'Location is required for onsite/hybrid work';
    }

    if (formData.age) {
      const ageNum = Number(formData.age);
      if (isNaN(ageNum) || ageNum < 18 || ageNum > 30) {
        newErrors.age = 'Age must be between 18 and 30';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData]);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit(formData);
    }
  }, [formData, validateForm, onSubmit]);

  // Memoize components to prevent unnecessary re-renders
  const FormSection = useMemo(() => React.memo(({
    icon: Icon,
    title,
    children
  }: {
    icon: React.ElementType;
    title: string;
    children: React.ReactNode;
  }) => (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700 hover:shadow-xl transition-all duration-300">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
          <Icon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h3>
      </div>
      {children}
    </div>
  )), []);

  const InputField = useMemo(() => React.memo(({
    label,
    name,
    type = 'text',
    required = false,
    error,
    children
  }: {
    label: string;
    name: string;
    type?: string;
    required?: boolean;
    error?: string;
    children?: React.ReactNode;
  }) => (
    <div className="space-y-2">
      <label
        htmlFor={name}
        className="block text-sm font-medium text-gray-700 dark:text-gray-300"
      >
        {label} {required && <span className="text-red-500">*</span>}
      </label>
      {children}
      {error && (
        <p className="text-sm text-red-600 dark:text-red-400 flex items-center gap-1">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
          {error}
        </p>
      )}
    </div>
  )), []);

  return (
    <form className="max-w-4xl mx-auto" onSubmit={handleSubmit}>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Find Your Perfect Internship Match
        </h2>
        <p className="text-gray-600 dark:text-gray-400 text-lg">
          Our AI-powered system with collaborative filtering will find the best opportunities for you
        </p>
      </div>

      <div className="space-y-6">
        {/* Skills Section */}
        <FormSection icon={Briefcase} title="Your Skills & Experience">
          <InputField label="Skills" name="skills" required error={errors.skills}>
            <textarea
              id="skills"
              name="skills"
              value={formData.skills}
              onChange={handleChange}
              rows={4}
              placeholder="e.g., Python, JavaScript, React, Data Analysis, Digital Marketing"
              className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white resize-none transition-all duration-200 ${
                errors.skills
                  ? 'border-red-500 focus:ring-red-500'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
            />
          </InputField>
        </FormSection>

        {/* Location & Work Preferences */}
        <FormSection icon={MapPin} title="Work Preferences">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <InputField label="Work Mode" name="workMode">
              <select
                id="workMode"
                name="workMode"
                value={formData.workMode}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="remote">Remote</option>
                <option value="onsite">Onsite</option>
                <option value="hybrid">Hybrid</option>
              </select>
            </InputField>

            {formData.workMode !== 'remote' && (
              <InputField label="Preferred Location" name="location" error={errors.location}>
                {locations.length > 0 ? (
                  <select
                    id="location"
                    name="location"
                    value={formData.location}
                    onChange={handleChange}
                    className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${
                      errors.location
                        ? 'border-red-500 focus:ring-red-500'
                        : 'border-gray-300 dark:border-gray-600'
                    }`}
                  >
                    <option value="">Select location</option>
                    {locations.map(loc => (
                      <option key={loc} value={loc}>
                        {loc}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    id="location"
                    name="location"
                    value={formData.location}
                    onChange={handleChange}
                    placeholder="e.g., New Delhi, Mumbai, Bangalore"
                    className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${
                      errors.location
                        ? 'border-red-500 focus:ring-red-500'
                        : 'border-gray-300 dark:border-gray-600'
                    }`}
                  />
                )}
              </InputField>
            )}
          </div>
        </FormSection>

        {/* Personal Details */}
        <FormSection icon={User} title="Personal Details">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <InputField label="Age" name="age" error={errors.age}>
              <input
                type="number"
                id="age"
                name="age"
                value={formData.age}
                onChange={handleChange}
                min="18"
                max="30"
                placeholder="22"
                className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${
                  errors.age
                    ? 'border-red-500 focus:ring-red-500'
                    : 'border-gray-300 dark:border-gray-600'
                }`}
              />
            </InputField>

            <InputField label="Duration Preference" name="duration">
              <select
                id="duration"
                name="duration"
                value={formData.duration}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="">Any duration</option>
                <option value="1 month">1 Month</option>
                <option value="2 months">2 Months</option>
                <option value="3 months">3 Months</option>
                <option value="6 months">6 Months</option>
                <option value="12 months">12 Months</option>
              </select>
            </InputField>

            <InputField label="Gender Preference" name="gender">
              <select
                id="gender"
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="any">Any</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </InputField>
          </div>
        </FormSection>

        {/* Financial Preferences */}
        <FormSection icon={DollarSign} title="Financial Preferences">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <InputField label="Family Income Range" name="familyIncome">
              <select
                id="familyIncome"
                name="familyIncome"
                value={formData.familyIncome}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="">Prefer not to say</option>
                <option value="0-100000">₹0 - ₹1 lac</option>
                <option value="100000-300000">₹1 lac - ₹3 lacs</option>
                <option value="300000-600000">₹3 lacs - ₹6 lacs</option>
                <option value="600000-800000">₹6 lacs - ₹8 lacs</option>
                <option value="800000+">₹8 lacs+</option>
              </select>
            </InputField>

            <InputField label="Payment Preference" name="paymentPreference">
              <select
                id="paymentPreference"
                name="paymentPreference"
                value={formData.paymentPreference}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="any">Any</option>
                <option value="paid">Paid only</option>
                <option value="unpaid">Unpaid acceptable</option>
              </select>
            </InputField>
          </div>
        </FormSection>

        {/* Submit Button */}
        <div className="flex justify-center pt-6">
          <button
            type="submit"
            disabled={isLoading}
            className="group relative px-12 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold text-lg rounded-2xl transition-all duration-300 transform hover:scale-105 hover:shadow-2xl disabled:scale-100 disabled:shadow-none min-w-[200px]"
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-3">
                <svg
                  className="animate-spin h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Finding Perfect Matches...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                Find My Internships
                <svg
                  className="w-5 h-5 group-hover:translate-x-1 transition-transform"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 7l5 5m0 0l-5 5m5-5H6"
                  />
                </svg>
              </span>
            )}
          </button>
        </div>

        {/* AI Features Badge */}
        <div className="text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-100 to-blue-100 dark:from-green-900/20 dark:to-blue-900/20 rounded-full text-sm font-medium text-gray-700 dark:text-gray-300">
            <svg
              className="w-4 h-4 text-green-600 dark:text-green-400"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
            Powered by AI & Machine Learning
          </div>
        </div>
      </div>
    </form>
  );
});

RecommendationForm.displayName = 'RecommendationForm';