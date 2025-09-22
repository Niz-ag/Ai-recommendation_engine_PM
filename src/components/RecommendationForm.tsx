import React, { useState, useCallback } from 'react';
import { User, MapPin, Briefcase, DollarSign, List, XCircle } from 'lucide-react';

// --- Helper Components (defined outside for stability) ---

interface InputFieldProps {
  label: string;
  name: string;
  required?: boolean;
  error?: string;
  children: React.ReactNode;
}

const InputField: React.FC<InputFieldProps> = React.memo(({ label, name, required, error, children }) => (
  <div className="space-y-2">
    <label htmlFor={name} className="block text-sm font-medium text-gray-700 dark:text-gray-300">
      {label} {required && <span className="text-red-500">*</span>}
    </label>
    {children}
    {error && (
      <p className="text-sm text-red-600 dark:text-red-400 flex items-center gap-1">
        <XCircle className="w-4 h-4" />
        {error}
      </p>
    )}
  </div>
));

interface FormSectionProps {
  icon: React.ElementType;
  title: string;
  children: React.ReactNode;
}

const FormSection: React.FC<FormSectionProps> = React.memo(({ icon: Icon, title, children }) => (
  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-100 dark:border-gray-700 hover:shadow-xl transition-all duration-300">
    <div className="flex items-center gap-3 mb-4">
      <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
        <Icon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
      </div>
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h3>
    </div>
    {children}
  </div>
));

// --- Main Recommendation Form Component ---

interface UserFormData {
  age: string;
  familyIncome: string;
  skills: string;
  location: string;
  workMode: string;
  paymentPreference: string;
  topN: number;
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
    paymentPreference: 'any',
    topN: 25,
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleChange = useCallback((
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'topN' ? parseInt(value, 10) : value
    }));

    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
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
      if (isNaN(ageNum) || ageNum < 21 || ageNum > 24) {
        newErrors.age = 'Eligibility requirement failed: Age must be between 21 and 24.';
      }
    } else {
        newErrors.age = 'Age is a required field for eligibility.';
    }
    
    if (formData.familyIncome === '800000+') {
      newErrors.familyIncome = 'Eligibility requirement failed: Income exceeds the scheme limit.';
    } else if (!formData.familyIncome) {
        newErrors.familyIncome = 'Family income is a required field for eligibility.';
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

  return (
    <form className="max-w-4xl mx-auto" onSubmit={handleSubmit}>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Find Your Perfect Internship Match
        </h2>
        
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
                errors.skills ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
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
              <InputField label="Preferred Location" name="location" required error={errors.location}>
                {locations.length > 0 ? (
                  <select
                    id="location"
                    name="location"
                    value={formData.location}
                    onChange={handleChange}
                    className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${
                      errors.location ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
                    }`}
                  >
                    <option value="">Select location</option>
                    {locations.map(loc => (<option key={loc} value={loc}>{loc}</option>))}
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
                      errors.location ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
                    }`}
                  />
                )}
              </InputField>
            )}
          </div>
        </FormSection>

        {/* Eligibility Details */}
        <FormSection icon={User} title="Eligibility Details">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <InputField label="Age" name="age" required error={errors.age}>
                  <input
                    type="number"
                    id="age"
                    name="age"
                    value={formData.age}
                    onChange={handleChange}
                    placeholder="22"
                    className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${
                      errors.age ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
                    }`}
                  />
                </InputField>
                <InputField label="Family Income Range" name="familyIncome" required error={errors.familyIncome}>
                  <select
                    id="familyIncome"
                    name="familyIncome"
                    value={formData.familyIncome}
                    onChange={handleChange}
                    className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white ${
                        errors.familyIncome ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
                    }`}
                  >
                    <option value="">Select income</option>
                    <option value="0-100000">₹0 - ₹1 lac</option>
                    <option value="100000-300000">₹1 lac - ₹3 lacs</option>
                    <option value="300000-600000">₹3 lacs - ₹6 lacs</option>
                    <option value="600000-800000">₹6 lacs - ₹8 lacs</option>
                    <option value="800000+">₹8 lacs+</option>
                  </select>
                </InputField>
            </div>
        </FormSection>

        {/* Other Preferences */}
        <FormSection icon={DollarSign} title="Other Preferences">
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
        </FormSection>

        {/* Display Options */}
        <FormSection icon={List} title="Display Options">
            <InputField label="Number of Results" name="topN">
                <select
                    id="topN"
                    name="topN"
                    value={formData.topN}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                >
                    <option value={10}>10</option>
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                </select>
            </InputField>
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
                <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                Finding Perfect Matches...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                Find My Internships
                <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
              </span>
            )}
          </button>
        </div>
      </div>
    </form>
  );
});

RecommendationForm.displayName = 'RecommendationForm';