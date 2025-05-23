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
    <footer className="footer"> 
      <div className="footer-content">
        <p className="copyright"><small>Copyright © 2025 PylibXAI, Maciej Falkowski</small></p>
        <div className="social-icons">
          {socialIcons.map(({ icon, url }) => (
            <a key={url} href={url} className="social-link">
              <FontAwesomeIcon 
                icon={icon} 
                className="social-icon"
              />
            </a>
          ))}
        </div>
      </div>
    </footer>
  )
}

export default Footer 