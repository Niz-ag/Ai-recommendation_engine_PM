import React, { useState, useCallback } from 'react';
import { Plus, CheckCircle, XCircle, Building, MapPin, DollarSign, FileText } from 'lucide-react';

interface InternshipData {
  title: string;
  company: string;
  location: string;
  skills: string;
  duration: string;
  stipend: string;
  description: string;
  gender: string;
}

const API_BASE_URL = `${window.location.protocol}//${window.location.hostname}:5000`;

// Memoized Input Field
const InputField = React.memo(
  ({
    label,
    name,
    required = false,
    error,
    children,
  }: {
    label: string;
    name: string;
    required?: boolean;
    error?: string;
    children: React.ReactNode;
  }) => (
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
  )
);

// Memoized Form Section
const FormSection = React.memo(
  ({ icon: Icon, title, children }: { icon: React.ElementType; title: string; children: React.ReactNode }) => (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
          <Icon className="w-5 h-5 text-orange-600 dark:text-orange-400" />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h3>
      </div>
      {children}
    </div>
  )
);

export const AdminPanel: React.FC = React.memo(() => {
  const [formData, setFormData] = useState<InternshipData>({
    title: '',
    company: '',
    location: '',
    skills: '',
    duration: '',
    stipend: '',
    description: '',
    gender: 'any',
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Memoized change handler
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
      const { name, value } = e.target;
      setFormData(prev => ({ ...prev, [name]: value }));

      if (errors[name]) {
        setErrors(prev => ({ ...prev, [name]: '' }));
      }
    },
    [errors]
  );

  // Validate form
  const validateForm = useCallback((): boolean => {
    const newErrors: Record<string, string> = {};
    if (!formData.title.trim()) newErrors.title = 'Title is required';
    if (!formData.company.trim()) newErrors.company = 'Company name is required';
    if (!formData.location.trim()) newErrors.location = 'Location is required';
    if (!formData.skills.trim()) newErrors.skills = 'Skills are required';
    if (!formData.duration) newErrors.duration = 'Duration is required';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData]);

  // Memoized submit handler
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!validateForm()) return;

      setIsSubmitting(true);
      setMessage(null);

      try {
        const response = await fetch(`${API_BASE_URL}/internships`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData),
        });

        const result = await response.json();

        if (response.ok) {
          setMessage({ type: 'success', text: result.message || 'Internship added successfully!' });
          setFormData({
            title: '',
            company: '',
            location: '',
            skills: '',
            duration: '',
            stipend: '',
            description: '',
            gender: 'any',
          });
        } else {
          setMessage({ type: 'error', text: result.error || 'Failed to add internship' });
        }
      } catch (error) {
        setMessage({ type: 'error', text: 'Network error. Please try again.' });
        console.error('Error adding internship:', error);
      } finally {
        setIsSubmitting(false);
      }
    },
    [formData, validateForm]
  );

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-orange-500 to-red-500 rounded-full mb-4">
          <Plus className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Add New Internship</h1>
        <p className="text-gray-600 dark:text-gray-400 text-lg">
          Contribute to the internship database and help students find opportunities
        </p>
      </div>

      {/* Success/Error Message */}
      {message && (
        <div
          className={`p-4 rounded-xl border ${
            message.type === 'success'
              ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
              : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
          }`}
        >
          <div className="flex items-center gap-3">
            {message.type === 'success' ? (
              <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
            ) : (
              <XCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
            )}
            <p
              className={`font-medium ${
                message.type === 'success'
                  ? 'text-green-800 dark:text-green-400'
                  : 'text-red-800 dark:text-red-400'
              }`}
            >
              {message.text}
            </p>
          </div>
        </div>
      )}

      {/* Form */}
      <form className="space-y-6" onSubmit={handleSubmit}>
        {/* Basic Information */}
        <FormSection icon={Building} title="Basic Information">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <InputField label="Internship Title" name="title" required error={errors.title}>
              <input
                type="text"
                name="title"
                value={formData.title}
                onChange={handleChange}
                placeholder="e.g., Software Development Intern"
                className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white transition-all duration-200 ${
                  errors.title ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
                }`}
              />
            </InputField>

            <InputField label="Company Name" name="company" required error={errors.company}>
              <input
                type="text"
                name="company"
                value={formData.company}
                onChange={handleChange}
                placeholder="e.g., Tech Solutions Inc."
                className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white transition-all duration-200 ${
                  errors.company ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
                }`}
              />
            </InputField>
          </div>
        </FormSection>

        {/* Location & Work Details */}
        <FormSection icon={MapPin} title="Location & Work Details">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <InputField label="Location" name="location" required error={errors.location}>
              <input
                type="text"
                name="location"
                value={formData.location}
                onChange={handleChange}
                placeholder="e.g., Mumbai, Remote, Hybrid"
                className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white transition-all duration-200 ${
                  errors.location ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
                }`}
              />
            </InputField>

            <InputField label="Duration" name="duration" required error={errors.duration}>
              <select
                name="duration"
                value={formData.duration}
                onChange={handleChange}
                className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white transition-all duration-200 ${
                  errors.duration ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
                }`}
              >
                <option value="">Select Duration</option>
                <option value="1 Month">1 Month</option>
                <option value="2 Months">2 Months</option>
                <option value="3 Months">3 Months</option>
                <option value="6 Months">6 Months</option>
                <option value="12 Months">12 Months</option>
              </select>
            </InputField>
          </div>
        </FormSection>

        {/* Skills & Requirements */}
        <FormSection icon={FileText} title="Skills & Requirements">
          <InputField label="Required Skills" name="skills" required error={errors.skills}>
            <textarea
              name="skills"
              value={formData.skills}
              onChange={handleChange}
              rows={3}
              placeholder="e.g., Python, JavaScript, React, Database Management"
              className={`w-full px-4 py-3 border rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white resize-none transition-all duration-200 ${
                errors.skills ? 'border-red-500 focus:ring-red-500' : 'border-gray-300 dark:border-gray-600'
              }`}
            />
          </InputField>
        </FormSection>

        {/* Additional Details */}
        <FormSection icon={DollarSign} title="Additional Details">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <InputField label="Stipend" name="stipend">
              <input
                name="stipend"
                value={formData.stipend}
                onChange={handleChange}
                placeholder="e.g., ₹15000/month, Unpaid, Performance based"
                className="w-full px-4 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white transition-all duration-200"
              />
            </InputField>

            <InputField label="Gender Preference" name="gender">
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white transition-all duration-200"
              >
                <option value="any">Any</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </InputField>
          </div>

          <div className="mt-6">
            <InputField label="Description" name="description">
              <textarea
                name="description"
                value={formData.description}
                onChange={handleChange}
                rows={4}
                placeholder="Brief description of the internship role, responsibilities, and what the intern will learn..."
                className="w-full px-4 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white resize-none transition-all duration-200"
              />
            </InputField>
          </div>
        </FormSection>

        {/* Submit Button */}
        <div className="flex justify-center pt-6">
          <button
            type="submit"
            disabled={isSubmitting}
            className="group relative px-12 py-4 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold text-lg rounded-2xl transition-all duration-300 transform hover:scale-105 hover:shadow-2xl disabled:scale-100 disabled:shadow-none min-w-[200px]"
          >
            {isSubmitting ? (
              <span className="flex items-center justify-center gap-3 animate-pulse">
                Adding Internship...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <Plus className="w-5 h-5" />
                Add Internship
              </span>
            )}
          </button>
        </div>
      </form>

      {/* Info Note */}
      <div className="text-center p-6 bg-gradient-to-r from-orange-50 to-red-50 dark:from-gray-800 dark:to-gray-700 rounded-2xl">
        <Building className="w-8 h-8 text-orange-600 dark:text-orange-400 mx-auto mb-3" />
        <p className="text-gray-600 dark:text-gray-300 font-medium">
          Thank you for contributing to the internship database! Your submission will be immediately available for students to discover.
        </p>
      </div>
    </div>
  );
});
