import React from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { 
  faYoutube,
  faGithub,
  faXTwitter,
  faDiscord
} from '@fortawesome/free-brands-svg-icons'
import './Footer.css'

function Footer() {
  const socialIcons = [
    { icon: faGithub, url: 'https://github.com/m-falkowski/pylibxai', label: 'Github' },
    { icon: faYoutube, url: 'https://youtube.com/@pylibxai', label: 'YouTube' },
    { icon: faXTwitter, url: 'https://x.com/pylibxai', label: 'X (Twitter)' },
    { icon: faDiscord, url: 'https://discord.gg/pylibxai', label: 'Discord' }
  ]

  return (
    <footer className="footer"> 
      <div className="footer-content">
        <p className="copyright"><small>Copyright © 2025 PylibXAI, Maciej Falkowski</small></p>
        <div className="social-icons">
          {socialIcons.map(({ icon, url, label }) => (
            <a 
              key={url} 
              href={url} 
              className="social-link"
              target="_blank"
              rel="noopener noreferrer"
              aria-label={label}
            >
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