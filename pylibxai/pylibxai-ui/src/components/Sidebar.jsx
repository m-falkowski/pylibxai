import React from 'react';
import { Nav } from 'react-bootstrap';
import './Sidebar.css';

function Sidebar() {
  return (
    <div className="sidebar bg-light" style={{ width: '250px', minHeight: '100vh', padding: '1rem' }}>
      <Nav className="flex-column">
        <Nav.Link href="#model-info" className="d-flex align-items-center">
          <i className="fas fa-info-circle me-2"></i>
          Model Information
        </Nav.Link>
        <Nav.Link href="#shap" className="d-flex align-items-center">
          <i className="fas fa-chart-bar me-2"></i>
          SHAP
        </Nav.Link>
      </Nav>
    </div>
  );
}

export default Sidebar; 