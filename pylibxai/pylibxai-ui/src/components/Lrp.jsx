import React, { useEffect, useRef, useState } from 'react'
import Chart from 'chart.js/auto'
import WaveSurfer from 'wavesurfer.js'
import './Lrp.css'

function Lrp() {
  const chartRef = useRef(null)
  const chartInstance = useRef(null)
  const waveformRef = useRef(null)
  const wavesurfer = useRef(null)
  const [statsCollapsed, setStatsCollapsed] = useState(false)
  const [audioCollapsed, setAudioCollapsed] = useState(false)
  const [imageCollapsed, setImageCollapsed] = useState(false)
  const [statsAnimating, setStatsAnimating] = useState(false)
  const [audioAnimating, setAudioAnimating] = useState(false)
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
    } else if (type === 'image') {
      setImageAnimating(true)
      setTimeout(() => {
        setImageCollapsed(!imageCollapsed)
        setImageAnimating(false)
      }, 300)
    }
  }

  useEffect(() => {
    // Fetch lrp attributions data
    const fetchAttributions = async () => {
      try {
        setIsLoading(true)
        setError(null)
        
        const response = await fetch(`${staticBaseUrl}/lrp/lrp_attributions.json`)
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`)
        }
        
        const data = await response.json()
        
        if (!data.attributions || !Array.isArray(data.attributions)) {
          throw new Error('Data is not in the expected format (object with attributions array)')
        }
        
        setAttributions(data.attributions)
        setIsLoading(false)
      } catch (error) {
        console.error("Error fetching attributions:", error)
        setError(error.message)
        setIsLoading(false)
      }
    }

    fetchAttributions()
  }, [staticBaseUrl])

  useEffect(() => {
    if (chartInstance.current) {
      chartInstance.current.destroy()
    }

    if (isLoading || attributions.length === 0) {
      return
    }

    const ctx = chartRef.current.getContext('2d')
    
    // Generate indices as labels (0 to attributions.length-1)
    const labels = Array.from({ length: attributions.length }, (_, i) => i)
    
    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: `LRP Attribution Values`,
          data: attributions,
          fill: false,
          borderColor: 'rgb(42, 123, 198)',
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'LRP Attributions'
          },
          tooltip: {
            callbacks: {
              title: (tooltipItems) => {
                return `Frame: ${tooltipItems[0].label}`
              },
              label: (tooltipItem) => {
                return `Value: ${tooltipItem.raw.toFixed(4)}`
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Frame Index'
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Attribution Value'
            }
          }
        }
      }
    })

    // Set chart container height
    if (chartRef.current) {
      chartRef.current.parentElement.style.height = '400px'
    }

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

    const loadAudio = async () => {
      try {
        await wavesurfer.current.load(`${staticBaseUrl}/input.wav`)
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
    }
  }, [attributions, isLoading, staticBaseUrl])

  return (
    <>
        <section className="mb-5">
          <div className="section-header">
            <h2 className="fw-bolder">Original audio</h2>
            <button 
              className="collapse-toggle"
              onClick={() => handleCollapse('audio')}
              style={{ transform: audioCollapsed ? 'rotate(180deg)' : 'rotate(0deg)' }}
            >
              {audioCollapsed ? '▼' : '▲'}
            </button>
          </div>
          <div className={`waveform-container ${audioCollapsed ? 'd-none' : ''} ${audioAnimating ? (audioCollapsed ? 'collapsing' : 'expanding') : ''}`}>
            <p>Audio waveform visualization with playback controls</p>
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
        
        <section className="mb-5">
          <div className="section-header">
            <h2 className="fw-bolder">LRP Attribution Image</h2>
            <div className="d-flex align-items-center gap-2">
              <button 
                className="collapse-toggle"
                onClick={() => handleCollapse('image')}
                style={{ transform: imageCollapsed ? 'rotate(180deg)' : 'rotate(0deg)' }}
              >
                {imageCollapsed ? '▼' : '▲'}
              </button>
            </div>
          </div>
          <div className={`image-container ${imageCollapsed ? 'd-none' : ''} ${imageAnimating ? (imageCollapsed ? 'collapsing' : 'expanding') : ''}`}>
            <p>Visual representation of LRP attributions</p>
            <img 
              src={`${staticBaseUrl}/lrp/${showAttribution ? 'lrp_attribution_heat_map.png' : 'lrp_spectogram.png'}`} 
              alt={showAttribution ? "LRP Attribution Heatmap" : "Audio Spectrogram"}
              className="lrp-image"
            />
              <button 
                className="view-toggle"
                onClick={() => setShowAttribution(!showAttribution)}
                title={showAttribution ? "Show Spectrogram" : "Show Attribution Heatmap"}
              >
                {showAttribution ? "Show Spectrogram" : "Show Attribution"}
              </button>
          </div>
        </section>

        <section className="mb-5">
          <div className="section-header">
            <h2 className="fw-bolder">LRP Attributions</h2>
            <button 
              className="collapse-toggle"
              onClick={() => handleCollapse('stats')}
              style={{ transform: statsCollapsed ? 'rotate(180deg)' : 'rotate(0deg)' }}
            >
              {statsCollapsed ? '▼' : '▲'}
            </button>
          </div>
          <div className={`chart-container ${statsCollapsed ? 'd-none' : ''} ${statsAnimating ? (statsCollapsed ? 'collapsing' : 'expanding') : ''}`}>
            {isLoading ? (
              <p>Loading LRP attributions data...</p>
            ) : error ? (
              <p className="text-danger">Error: {error}</p>
            ) : (
              <>
                <p>LRP attributions visualization showing the importance of each frame in the audio sample.</p>
                <p><small>Total frames: {attributions.length}</small></p>
                <canvas ref={chartRef}></canvas>
              </>
            )}
          </div>
        </section>

    </>
  )
}

export default Lrp
