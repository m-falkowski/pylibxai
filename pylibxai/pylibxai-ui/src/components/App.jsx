import React, { useState } from 'react';
import Sidebar from './Sidebar';
import ModelInfo from './ModelInfo';
import SHAP from './SHAP';
import LIME from './LIME';
import LRP from './LRP';
import InferenceInfo from './InferenceInfo';
import './Main.css';

function App() {
  const [currentSection, setCurrentSection] = useState('model-info');

  const handleNavigation = (section) => {
    setCurrentSection(section);
  };

  const renderContent = () => {
    switch (currentSection) {
      case 'model-info':
        return <ModelInfo />;
      case 'shap':
        return <SHAP />;
      case 'lime':
        return <LIME />;
      case 'lrp':
        return <LRP />;
      case 'inference':
        return <InferenceInfo />;
      default:
        return <ModelInfo />;
    }
  };

  return (
    <div className="app">
      <Sidebar onNavigate={handleNavigation} currentSection={currentSection} />
      {renderContent()}
    </div>
  );
}

export default App; 