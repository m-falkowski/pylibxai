import React, { useState, useContext } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faBell, faCircleInfo } from '@fortawesome/free-solid-svg-icons'
import './Navbar.css'
import { NotificationContext } from '../App'

function Navbar() {
  const [showAlerts, setShowAlerts] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const { notifications } = useContext(NotificationContext);

  const softwareInfo = {
    name: 'PylibXAI',
    version: '1.0.0',
    copyright: '© 2025 Maciej Falkowski',
    description: 'Explainable AI Library for Python',
    repository: 'https://github.com/m-falkowski/pylibxai'
  };

  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container-fluid p-0">
        <div className="d-flex align-items-center" style={{ width: '250px', padding: '1rem', justifyContent: 'center' }}>
          <a href="/" className="logo-text">
            PylibXAI
          </a>
        </div>

        <div className="navbar-icons">
          <button 
            className="icon-button"
            onClick={() => setShowAlerts(!showAlerts)}
            aria-label="Show alerts"
          >
            <FontAwesomeIcon icon={faBell} />
            {notifications.length > 0 && <span className="notification-badge">{notifications.length}</span>}
          </button>

          <button 
            className="icon-button"
            onClick={() => setShowInfo(!showInfo)}
            aria-label="Show software information"
          >
            <FontAwesomeIcon icon={faCircleInfo} />
          </button>

          {showAlerts && (
            <div className="dropdown-panel alerts-panel">
              <h3>Notifications</h3>
              {notifications.length === 0 ? (
                <div className="alert-item info">
                  <span className="alert-message">No notifications</span>
                </div>
              ) : notifications.map(alert => (
                <div key={alert.id} className={`alert-item ${alert.type || 'info'}`}>
                  <span className="alert-message">{alert.message}</span>
                </div>
              ))}
            </div>
          )}

          {showInfo && (
            <div className="dropdown-panel info-panel">
              <h3>{softwareInfo.name}</h3>
              <div className="info-item">
                <strong>Version:</strong> {softwareInfo.version}
              </div>
              <div className="info-item">
                <strong>Copyright:</strong> {softwareInfo.copyright}
              </div>
              <div className="info-item">
                <strong>Description:</strong> {softwareInfo.description}
              </div>
              <div className="info-item">
                <a href={softwareInfo.repository} target="_blank" rel="noopener noreferrer">
                  View on GitHub
                </a>
              </div>
            </div>
          )}
        </div>
      </div>
    </nav>
  )
}

export default Navbar 