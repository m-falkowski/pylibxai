import React from 'react'
import './Footer.css'

function Footer() {
  return (
    <footer className="container-fluid bg-dark text-white mt-5"> 
      <div className="container text-center pt-3 pb-0"> 
        <p><small>Copyright © 2025 Jan Kowalski</small></p> 
      </div> 
      <div className="container text-center pt-0 pb-3"> 
        <a href="https://www.facebook.com"><i className="NavBarIcon fa-brands fa-facebook" style={{ color: '#B7C3F3' }}></i></a>
        <a href="https://twitter.com"><i className="NavBarIcon fa-brands fa-twitter" style={{ color: '#B7C3F3' }}></i></a>
        <a href="https://www.linkedin.com"><i className="NavBarIcon fa-brands fa-linkedin" style={{ color: '#B7C3F3' }}></i></a>
        <a href="https://www.instagram.com"><i className="NavBarIcon fa-brands fa-instagram" style={{ color: '#B7C3F3' }}></i></a>
      </div> 
    </footer>
  )
}

export default Footer 