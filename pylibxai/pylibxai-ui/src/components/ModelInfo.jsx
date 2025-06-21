import React, { useEffect, useState } from 'react'
import './ModelInfo.css'

function ModelInfo() {
  const [labels, setLabels] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  // Get the static file server port from environment variables
  const staticPort = import.meta.env.VITE_PYLIBXAI_STATIC_PORT || '9000'
  const staticBaseUrl = `http://localhost:${staticPort}`

  useEffect(() => {
    const fetchLabels = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const response = await fetch(`${staticBaseUrl}/labels.json`)
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`)
        }
        const data = await response.json()
        // Convert {"blues": 0, ...} to array of {id, name}
        const labelArray = Object.entries(data).map(([name, id]) => ({ id, name }))
        // Sort by id
        labelArray.sort((a, b) => a.id - b.id)
        setLabels(labelArray)
        setIsLoading(false)
      } catch (error) {
        setError(error.message)
        setIsLoading(false)
      }
    }
    fetchLabels()
  }, [staticBaseUrl])

  return (
    <>
        <div className="container-fluid text-center">
          <h1>Model Information</h1>
          <br/>
          <h3>Label Mapping</h3>
          {isLoading ? (
            <p>Loading labels...</p>
          ) : error ? (
            <p className="text-danger">Error: {error}</p>
          ) : (
            <div className="table-responsive d-flex justify-content-center">
              <table className="table table-bordered w-auto">
                <thead>
                  <tr>
                    <th>Class ID</th>
                    <th>Class Name</th>
                  </tr>
                </thead>
                <tbody>
                  {labels.map(label => (
                    <tr key={label.id}>
                      <td>{label.id}</td>
                      <td>{label.name}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
    </>
  )
}

export default ModelInfo
