import React from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faFacebook, 
  faTwitter, 
  faLinkedin, 
  faInstagram 
} from '@fortawesome/free-brands-svg-icons'
import { faMagnifyingGlass } from '@fortawesome/free-solid-svg-icons'
import './Navbar.css'

function Navbar() {
  const socialIcons = [
    { icon: faFacebook, name: 'facebook' },
    { icon: faTwitter, name: 'twitter' },
    { icon: faLinkedin, name: 'linkedin' },
    { icon: faInstagram, name: 'instagram' }
  ]

  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark sticky-top">
      <div className="container-fluid">
        {/* Social Media Links */}
        <div className="d-flex align-items-center">
          {socialIcons.map(({ icon, name }) => (
            <a 
              key={name}
              href={`https://www.${name}.com`} 
              className="me-3 text-white text-decoration-none"
            >
              <FontAwesomeIcon icon={icon} />
            </a>
          ))}
        </div>

        {/* Navigation Links */}
        <button 
          className="navbar-toggler" 
          type="button" 
          data-bs-toggle="collapse" 
          data-bs-target="#navbarNav" 
          aria-controls="navbarNav" 
          aria-expanded="false" 
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse justify-content-center" id="navbarNav">
          <ul className="navbar-nav">
            {['O MNIE', 'BLOG', 'KONTAKT', 'O PROJEKCIE'].map((item, index) => (
              <li key={index} className="nav-item">
                <a className="nav-link" href={`/${item.toLowerCase().replace(' ', '')}`}>
                  {item}
                </a>
              </li>
            ))}
          </ul>
        </div>

        {/* Search Bar */}
        <div className="d-flex align-items-center">
          <div className="input-group">
            <span className="input-group-text bg-dark border-0">
              <FontAwesomeIcon icon={faMagnifyingGlass} className="text-white" />
            </span>
            <input 
              type="text" 
              className="form-control bg-dark text-white border-0" 
              placeholder="Szukaj na blogu"
              aria-label="Search blog"
            />
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar 