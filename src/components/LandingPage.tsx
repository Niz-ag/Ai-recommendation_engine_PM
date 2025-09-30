import React from "react";
import { ArrowRight } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle"; 

interface LandingPageProps {
  onGetStarted: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted }) => {
  return (
    <div className="relative w-screen h-screen flex flex-col justify-between items-center px-4 overflow-hidden">
      <img
        src="/bkcgd-white.jpg"
        alt="Background Light"
        className="absolute inset-0 w-full h-full object-cover -z-10 dark:hidden"
      />
      <img
        src="/bckgd-black.png"
        alt="Background Dark"
        className="absolute inset-0 w-full h-full object-cover -z-10 hidden dark:block"
      />


      {/* Overlay for readability */}
      <div className="absolute inset-0 bg-black/10 dark:bg-black/50 -z-10" />

      {/* Dark Mode Toggle */}
      <div className="absolute top-6 right-6 z-20">
        <ThemeToggle />
      </div>

      {/* Heading Section */}
      <div className="flex-1 flex items-center justify-center text-center max-w-3xl">
        <h1 className="text-5xl md:text-7xl font-bold leading-tight">
  <span className="block text-[#333232] dark:text-[#BDB9B9]">
    Find Your Perfect Internship Under
  </span>
  <span
    className="block text-transparent bg-clip-text h-100"
    style={{
      backgroundImage: "linear-gradient(to right, #FF9933 30%, #138808 70%) ",
    }}
  >
    PM Internship Scheme
  </span>
</h1>
    
      </div>

      {/* Get Started Button */} 
      <div className="mb-12 ">
        <button 
          onClick={onGetStarted}
      className="
  group inline-flex items-center gap-3
  bg-gray-600 dark:bg-gray-200
  text-white dark:text-gray-900
  hover:bg-blue-700 dark:hover:bg-blue-600
  hover:text-white dark:hover:text-white
  font-semibold text-xl
  px-10 py-5
  rounded-2xl
  transition-all duration-300 transform
  hover:scale-105
  shadow-xl hover:shadow-2xl
"        >
          Get Started
          <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform duration-300" />
        </button>

      </div>

      {/* Universities Section */}
      <div className="pb-6 text-center">
        <p className="text-sm text-gray-700 dark:text-gray-300">
          Trusted by students from top universities
        </p>
      </div>
    </div>
  );
};
