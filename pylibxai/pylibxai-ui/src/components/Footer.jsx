import React from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { 
  faFacebook, 
  faTwitter, 
  faLinkedin, 
  faInstagram 
} from '@fortawesome/free-brands-svg-icons'
import './Footer.css'

function Footer() {
  const socialIcons = [
    { icon: faFacebook, url: 'https://www.facebook.com' },
    { icon: faTwitter, url: 'https://twitter.com' },
    { icon: faLinkedin, url: 'https://www.linkedin.com' },
    { icon: faInstagram, url: 'https://www.instagram.com' }
  ]

  return (
    <footer className="container-fluid bg-dark text-white mt-5"> 
      <div className="container text-center pt-3 pb-0"> 
        <p><small>Copyright © 2025 Jan Kowalski</small></p> 
      </div> 
      <div className="container text-center pt-0 pb-3"> 
        {socialIcons.map(({ icon, url }) => (
          <a key={url} href={url}>
            <FontAwesomeIcon 
              icon={icon} 
              className="NavBarIcon" 
              style={{ color: '#B7C3F3' }} 
            />
          </a>
        ))}
      </div> 
    </footer>
  )
}

export default Footer 