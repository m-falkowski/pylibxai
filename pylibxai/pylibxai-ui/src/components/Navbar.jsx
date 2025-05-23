import React from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import './Navbar.css'

function Navbar() {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container-fluid p-0">
        <div className="d-flex align-items-center" style={{ width: '250px', padding: '1rem', justifyContent: 'center' }}>
          <a href="/" className="logo-text">
            PylibXAI
          </a>
        </div>
      </div>
    </nav>
  )
}

export default Navbar 