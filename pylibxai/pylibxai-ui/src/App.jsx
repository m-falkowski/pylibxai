import 'bootstrap/dist/css/bootstrap.min.css'
import { library } from '@fortawesome/fontawesome-svg-core'
import { fas } from '@fortawesome/free-solid-svg-icons'
import './App.css'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import ContentPage from './components/ContentPage'
import Footer from './components/Footer'

// Add all solid icons to the library
library.add(fas)

import React, { useState } from 'react';

function App() {
  const [currentSection, setCurrentSection] = useState('model-info');

  const handleNavigation = (section) => {
    setCurrentSection(section);
  };

  return (
    <div className="app-container">
      <Navbar />
      <div className="main-layout">
        <Sidebar onNavigate={handleNavigation} currentSection={currentSection} />
        <div className="content-wrapper">
          <ContentPage section={currentSection} />
          <Footer />
        </div>
      </div>
    </div>
  );
}

export default App
