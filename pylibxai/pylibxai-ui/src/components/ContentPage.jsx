import React from 'react'
import Lime from './Lime'
import Igradients from './Igradients'
import Lrp from './Lrp'
import ModelInfo from './ModelInfo'
import './ContentPage.css'

const routes = {
  lime: Lime,
  igradients: Igradients,
  lrp: Lrp,
  'model-info': ModelInfo
}

function ContentPage({ section }) {
  const Component = routes[section]

  return (
    <div className="main-container">
      <div className="main-content">
        {Component && <Component />}
      </div>
    </div>
  )
}

export default ContentPage 