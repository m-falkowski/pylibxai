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

  useEffect(() => {
    if (chartInstance.current) {
      chartInstance.current.destroy()
    }

    const ctx = chartRef.current.getContext('2d')
    
    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: ['Styczeń', 'Luty', 'Marzec', 'Kwiecień', 'Maj', 'Czerwiec'],
        datasets: [{
          label: 'Liczba odwiedzin bloga',
          data: [65, 59, 80, 81, 56, 55],
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
            text: 'Statystyki odwiedzin bloga'
          }
        },
        scales: {
          y: {
            beginAtZero: true
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
  }, [])

  return (
    <>
        <section className="mb-5">
          <div className="section-header">
            <h2 className="fw-bolder">Statystyki bloga</h2>
            <button 
              className="collapse-toggle"
              onClick={() => setStatsCollapsed(!statsCollapsed)}
            >
              {statsCollapsed ? '▼' : '▲'}
            </button>
          </div>
          <div className={`chart-container ${statsCollapsed ? 'd-none' : ''}`}>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
            <canvas ref={chartRef}></canvas>
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
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat
            <div ref={waveformRef}></div>
            <div className="controls mt-3">
              <button 
                className="btn btn-primary me-2"
                onClick={() => wavesurfer.current.playPause()}
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
