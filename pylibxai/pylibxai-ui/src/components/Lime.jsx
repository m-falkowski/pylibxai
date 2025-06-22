import React, { useEffect, useRef, useState } from 'react'
import Chart from 'chart.js/auto'
import WaveSurfer from 'wavesurfer.js'
import './Lime.css'

function Lime() {
  const chartRef = useRef(null)
  const chartInstance = useRef(null)
  const waveformRef = useRef(null)
  const waveformRefOriginal = useRef(null)
  const wavesurfer = useRef(null)
  const wavesurferOriginal = useRef(null)
  const [statsCollapsed, setStatsCollapsed] = useState(false)
  const [audioCollapsed, setAudioCollapsed] = useState(false)
  const [originalCollapsed, setOriginalCollapsed] = useState(false)
  const [imageCollapsed, setImageCollapsed] = useState(false)
  const [statsAnimating, setStatsAnimating] = useState(false)
  const [audioAnimating, setAudioAnimating] = useState(false)
  const [originalAnimating, setOriginalAnimating] = useState(false)
  const [imageAnimating, setImageAnimating] = useState(false)
  const [showAttribution, setShowAttribution] = useState(true)
  const [attributions, setAttributions] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)
  
  // Get the static file server port from environment variables
  const staticPort = import.meta.env.VITE_PYLIBXAI_STATIC_PORT || '9000'
  const staticBaseUrl = `http://localhost:${staticPort}`

  const handleCollapse = (type) => {
    if (type === 'stats') {
      setStatsAnimating(true)
      setTimeout(() => {
        setStatsCollapsed(!statsCollapsed)
        setStatsAnimating(false)
      }, 300)
    } else if (type === 'audio') {
      setAudioAnimating(true)
      setTimeout(() => {
        setAudioCollapsed(!audioCollapsed)
        setAudioAnimating(false)
      }, 300)
    } else if (type === 'original') {
      setOriginalAnimating(true)
      setTimeout(() => {
        setOriginalCollapsed(!originalCollapsed)
        setOriginalAnimating(false)
      }, 300)
    } else if (type === 'image') {
      setImageAnimating(true)
      setTimeout(() => {
        setImageCollapsed(!imageCollapsed)
        setImageAnimating(false)
      }, 300)
    }
  }

  useEffect(() => {
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#4F4A85',
      progressColor: '#383351',
      cursorColor: '#383351',
      barWidth: 2,
      barRadius: 3,
      responsive: true,
      height: 100,
      barGap: 3
    })
    
    wavesurferOriginal.current = WaveSurfer.create({
      container: waveformRefOriginal.current,
      waveColor: '#4F4A85',
      progressColor: '#383351',
      cursorColor: '#383351',
      barWidth: 2,
      barRadius: 3,
      responsive: true,
      height: 100,
      barGap: 3
    })

    const loadAudio = async () => {
      try {
        await wavesurfer.current.load(`${staticBaseUrl}/lime/explanation.wav`)
      } catch (error) {
        console.error('Failed to load audio:', error)
      }
      
      try {
        await wavesurferOriginal.current.load(`${staticBaseUrl}/input.wav`)
      } catch (error) {
        console.error('Failed to load audio:', error)
      }
    }
    loadAudio()

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy()
      }
      if (wavesurfer.current) {
        wavesurfer.current.pause()
        wavesurfer.current.destroy()
      }
      if (wavesurferOriginal.current) {
        wavesurferOriginal.current.pause()
        wavesurferOriginal.current.destroy()
      }
    }
  }, [attributions, isLoading, staticBaseUrl])

  return (
    <>
        <section className="mb-5">
          <div className="section-header">
            <h2 className="fw-bolder">Original audio</h2>
            <button 
              className="collapse-toggle"
              onClick={() => handleCollapse('orignal')}
              style={{ transform: originalCollapsed ? 'rotate(180deg)' : 'rotate(0deg)' }}
            >
              {originalCollapsed ? '▼' : '▲'}
            </button>
          </div>
          <div className={`waveform-container ${originalCollapsed ? 'd-none' : ''} ${originalAnimating ? (originalCollapsed ? 'collapsing' : 'expanding') : ''}`}>
            <div ref={waveformRefOriginal}></div>
            <div className="controls mt-3">
              <button 
                className="btn btn-primary me-2"
                onClick={() => wavesurferOriginal.current && wavesurferOriginal.current.playPause()}
              >
                Play/Pause
              </button>
            </div>
          </div>
        </section>

        <section className="mb-5">
          <div className="section-header">
            <h2 className="fw-bolder">LIME explanation</h2>
            <button 
              className="collapse-toggle"
              onClick={() => handleCollapse('audio')}
              style={{ transform: audioCollapsed ? 'rotate(180deg)' : 'rotate(0deg)' }}
            >
              {audioCollapsed ? '▼' : '▲'}
            </button>
          </div>
          <div className={`waveform-container ${audioCollapsed ? 'd-none' : ''} ${audioAnimating ? (audioCollapsed ? 'collapsing' : 'expanding') : ''}`}>
            <p>LIME explanation for label: LABEL</p>
            <div ref={waveformRef}></div>
            <div className="controls mt-3">
              <button 
                className="btn btn-primary me-2"
                onClick={() => wavesurfer.current && wavesurfer.current.playPause()}
              >
                Play/Pause
              </button>
            </div>
          </div>
        </section>
    </>
  )
}

export default Lime 
