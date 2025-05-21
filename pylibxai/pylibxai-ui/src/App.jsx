import { useState } from 'react'
import 'bootstrap/dist/css/bootstrap.min.css'
import '@fortawesome/fontawesome-free/css/all.min.css'
import './App.css'
import Navbar from './components/Navbar'
import Footer from './components/Footer'

function App() {
  return (
    <div className="d-flex flex-column h-100">
      <Navbar />

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
            <p className="fs-5 mb-4">
              Pomimo, że moja praca to moja pasja, staram się znaleźć czas również na inne 
              zainteresowania. Lubię czytać książki o nowych technologiach oraz brać udział 
              w konferencjach i szkoleniach, które pozwalają mi na rozwijanie swoich umiejętności 
              i poznawanie najnowszych trendów w dziedzinie elektroniki.
            </p>
            <p className="fs-5 mb-4">
              Jednym z moich celów na przyszłość jest przekazywanie swojej wiedzy i doświadczenia 
              innym. Dlatego też postanowiłem stworzyć tego bloga, na którym będę dzielił się z 
              Wami moimi przemyśleniami oraz ciekawymi informacjami na temat projektowania układów 
              scalonych. Mam nadzieję, że będzie to dla Was źródło inspiracji oraz pozwoli Wam na 
              poszerzenie swoich horyzontów w tej dziedzinie.
            </p>
            <h2 className="fw-bolder mb-4 mt-5">Doświadczenie</h2>
            <p className="fs-5 mb-4">
              W mojej pracy stawiam na precyzję i dbałość o szczegóły.
              W projektowaniu układów scalonych każdy drobiazg ma znaczenie,
              dlatego zawsze staram się działać systematycznie i zgodnie z najwyższymi standardami.
              Jestem przekonany, że dzięki temu moje projekty są nie tylko funkcjonalne,
              ale również stabilne i bezpieczne.

              Posiadam doświadczenie zawodowe w:
              <ul className="fs-5 mb-4">
                <li>Lorem ipsum dolor sit amet</li>
                <li>Integer euismod lacus luctus magna</li>
                <li>Morbi lacinia molestie dui</li>
              </ul>
            </p>
          </section>
        </div>
      </div>

      <Footer />
    </div>
  )
}

export default App
