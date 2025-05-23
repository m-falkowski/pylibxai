import 'bootstrap/dist/css/bootstrap.min.css'
import { library } from '@fortawesome/fontawesome-svg-core'
import { fas } from '@fortawesome/free-solid-svg-icons'
import './App.css'
import Navbar from './components/Navbar'
import ContentPage from './components/ContentPage'
import Footer from './components/Footer'
import Sidebar from './components/Sidebar'

// Add all solid icons to the library
library.add(fas)

function App() {
  return (
    <div className="d-flex flex-column min-vh-100">
      <Navbar />
      <div className="d-flex flex-grow-1">
        <Sidebar />
        <div className="flex-grow-1 d-flex flex-column">
          <ContentPage />
          <Footer />
        </div>
      </div>
    </div>
  )
}

export default App
