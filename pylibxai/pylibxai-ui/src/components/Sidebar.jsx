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

function Sidebar() {
  return (
    <div className="sidebar bg-light">
      <Nav className="flex-column">
        <Nav.Link href="#model-info" className="d-flex align-items-center">
          <FontAwesomeIcon icon={faInfoCircle} className="me-2" />
          Model Information
        </Nav.Link>
        <Nav.Link href="#shap" className="d-flex align-items-center">
          <FontAwesomeIcon icon={faChartBar} className="me-2" />
          SHAP
        </Nav.Link>
        <Nav.Link href="#lime" className="d-flex align-items-center">
          <FontAwesomeIcon icon={faLightbulb} className="me-2" />
          LIME
        </Nav.Link>
        <Nav.Link href="#lrp" className="d-flex align-items-center">
          <FontAwesomeIcon icon={faBrain} className="me-2" />
          LRP
        </Nav.Link>
        <Nav.Link href="#inference" className="d-flex align-items-center">
          <FontAwesomeIcon icon={faCircleInfo} className="me-2" />
          Inference Information
        </Nav.Link>
      </Nav>
    </div>
  );
}

export default Sidebar; 