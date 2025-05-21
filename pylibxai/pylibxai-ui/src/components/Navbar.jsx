import React from 'react'
import './Navbar.css'

function Navbar() {
  return (
    <div className="navbar">
      <div className="topLeft">
        <a href="https://www.facebook.com"><i className="NavBarIcon fa-brands fa-facebook"></i></a>
        <a href="https://twitter.com"><i className="NavBarIcon fa-brands fa-twitter"></i></a>
        <a href="https://www.linkedin.com"><i className="NavBarIcon fa-brands fa-linkedin"></i></a>
        <a href="https://www.instagram.com"><i className="NavBarIcon fa-brands fa-instagram"></i></a>
      </div>
      <div className="topCenter">
        <ul className="NavBarList">
          <li><a href="/">O MNIE</a></li>
          <li><a href="/blog">BLOG</a></li>
          <li><a href="/contact">KONTAKT</a></li>
          <li><a href="/oprojekcie">O PROJEKCIE</a></li>
        </ul>
      </div>
      <div className="topRight">
        <i className="NavBarSearchIcon fa-solid fa-magnifying-glass"></i>
        <input 
          id="search" 
          type="text" 
          aria-label="Search blog"
          className="bg-gray w-50 search-field"
          placeholder="Szukaj na blogu"
        />
      </div>
    </div>
  )
}

export default Navbar 