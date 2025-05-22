import React from 'react'
import './Navbar.css'

function Navbar() {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark sticky-top">
      <div className="container-fluid">
        {/* Social Media Links */}
        <div className="d-flex align-items-center">
          {['facebook', 'twitter', 'linkedin', 'instagram'].map(social => (
            <a 
              key={social}
              href={`https://www.${social}.com`} 
              className="me-3 text-white text-decoration-none"
            >
              <i className={`fa-brands fa-${social}`}></i>
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
              <i className="fa-solid fa-magnifying-glass text-white"></i>
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