import React, { useEffect, useRef } from 'react'
import Chart from 'chart.js/auto'
import WaveSurfer from 'wavesurfer.js'
import './Main.css'

function Main() {
  const chartRef = useRef(null)
  const chartInstance = useRef(null)
  const waveformRef = useRef(null)
  const wavesurfer = useRef(null)

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

    wavesurfer.current.load('sandman_5s.wav')

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy()
      }
      if (wavesurfer.current) {
        wavesurfer.current.destroy()
      }
    }
  }, [])

  return (
    <div className="main-container">
      <div className="main-content">
        <div className="container-fluid text-center rounded-circle image-cropper">
          <img 
            src="/assets/face.jpg" 
            aria-label="Zdjęcie przedstawiające osobę ze skrzyżowanymi rękami" 
            className="rounded-circle circle-image" 
          />
        </div>
        <div className="container-fluid text-center">
          <h1>Blog Jan Kowalski</h1>
          <br/>
        </div>
        <section className="mb-5">
          <p className="fs-5 mb-4">
            Witajcie! Nazywam się Jan Kowalski i jestem absolwentem Politechniki Warszawskiej. 
            Od zawsze interesowałem się elektroniką i technologią, a w szczególności układami 
            scalonymi oraz procesorami.
          </p>
          <p className="fs-5 mb-4">
            Już podczas studiów zacząłem pracować przy projektowaniu układów scalonych, 
            a po ukończeniu studiów rozpocząłem swoją karierę jako inżynier ds. projektowania 
            układów scalonych. Od tamtego czasu rozwijam swoje umiejętności i zdobywam coraz 
            większe doświadczenie w tej dziedzinie.
          </p>
        </section>

        <section className="mb-5">
          <h2 className="fw-bolder mb-4">Statystyki bloga</h2>
          <div className="chart-container">
            <canvas ref={chartRef}></canvas>
          </div>
        </section>

        <section className="mb-5">
          <h2 className="fw-bolder mb-4">Audio Visualization</h2>
          <div className="waveform-container">
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
      </div>
    </div>
  )
}

export default Main 