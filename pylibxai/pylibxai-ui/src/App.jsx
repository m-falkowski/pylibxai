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

import React, { useState, createContext, useCallback } from 'react';

// Notification context
export const NotificationContext = createContext({
  notifications: [],
  pushNotification: () => {},
});

function App() {
  const [currentSection, setCurrentSection] = useState('model-info');
  const [notifications, setNotifications] = useState([]); // {id, type, message}
  const maxNotifications = 10;
  const handleNavigation = (section) => {
    setCurrentSection(section);
  };

  // Push notification (circular queue)
  const pushNotification = useCallback((notification) => {
    setNotifications(prev => {
      const next = [
        { id: Date.now() + Math.random(), ...notification },
        ...prev
      ];
      return next.slice(0, maxNotifications);
    });
  }, []);

  return (
    <NotificationContext.Provider value={{ notifications, pushNotification }}>
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
    </NotificationContext.Provider>
  );
}

export default App
