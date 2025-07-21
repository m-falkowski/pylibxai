import React from 'react';
import { Nav } from 'react-bootstrap';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faInfoCircle, 
  faChartBar, 
  faLightbulb, 
  faBrain,
  faCircleInfo 
} from '@fortawesome/free-solid-svg-icons';
import './Sidebar.css';

function Sidebar({ onNavigate, currentSection }) {
  return (
    <div className="sidebar bg-light">
      <Nav className="flex-column">
        <Nav.Link 
          href="#model-info" 
          className={`d-flex align-items-center${currentSection === 'model-info' ? ' active' : ''}`}
          onClick={() => onNavigate('model-info')}
        >
          <FontAwesomeIcon icon={faInfoCircle} className="me-2" />
          Model Information
        </Nav.Link>
        <Nav.Link 
          href="#igradients" 
          className={`d-flex align-items-center${currentSection === 'igradients' ? ' active' : ''}`}
          onClick={() => onNavigate('igradients')}
        >
          <FontAwesomeIcon icon={faChartBar} className="me-2" />
          Integrated gradients
        </Nav.Link>
        <Nav.Link 
          href="#lime" 
          className={`d-flex align-items-center${currentSection === 'lime' ? ' active' : ''}`}
          onClick={() => onNavigate('lime')}
        >
          <FontAwesomeIcon icon={faLightbulb} className="me-2" />
          LIME
        </Nav.Link>
        <Nav.Link 
          href="#lrp" 
          className={`d-flex align-items-center${currentSection === 'lrp' ? ' active' : ''}`}
          onClick={() => onNavigate('lrp')}
        >
          <FontAwesomeIcon icon={faBrain} className="me-2" />
          LRP
        </Nav.Link>
      </Nav>
    </div>
  );
}

export default Sidebar; 