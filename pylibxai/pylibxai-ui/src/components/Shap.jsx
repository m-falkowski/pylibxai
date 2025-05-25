import React, { useEffect, useRef, useState } from 'react'
import Chart from 'chart.js/auto'
import WaveSurfer from 'wavesurfer.js'
import './Shap.css'

function Shap() {
  const chartRef = useRef(null)
  const chartInstance = useRef(null)
  const waveformRef = useRef(null)
  const wavesurfer = useRef(null)
  const [statsCollapsed, setStatsCollapsed] = useState(false)
  const [audioCollapsed, setAudioCollapsed] = useState(false)
  const [attributions, setAttributions] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Fetch SHAP attributions data
    const fetchAttributions = async () => {
      try {
        setIsLoading(true)
        setError(null)
        // from http://localhost:9000/
        const response = await fetch(`http://localhost:9000/shap_attributions.json`)
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
  }, [])

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
          label: `SHAP Attribution Values`,
          data: attributions,
          fill: false,
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'SHAP Attributions'
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
        await wavesurfer.current.load('sandman_5s.wav')
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
  }, [attributions, isLoading])

  return (
    <>
        <section className="mb-5">
          <div className="section-header">
            <h2 className="fw-bolder">SHAP Attributions</h2>
            <button 
              className="collapse-toggle"
              onClick={() => setStatsCollapsed(!statsCollapsed)}
            >
              {statsCollapsed ? '▼' : '▲'}
            </button>
          </div>
          <div className={`chart-container ${statsCollapsed ? 'd-none' : ''}`}>
            {isLoading ? (
              <p>Loading SHAP attributions data...</p>
            ) : error ? (
              <p className="text-danger">Error: {error}</p>
            ) : (
              <>
                <p>SHAP attributions visualization showing the importance of each frame in the audio sample.</p>
                <p><small>Total frames: {attributions.length}</small></p>
                <canvas ref={chartRef}></canvas>
              </>
            )}
          </div>
        </section>

        <section className="mb-5">
          <div className="section-header">
            <h2 className="fw-bolder">Audio Visualization</h2>
            <button 
              className="collapse-toggle"
              onClick={() => setAudioCollapsed(!audioCollapsed)}
            >
              {audioCollapsed ? '▼' : '▲'}
            </button>
          </div>
          <div className={`waveform-container ${audioCollapsed ? 'd-none' : ''}`}>
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
    </>
  )
}

export default Shap
