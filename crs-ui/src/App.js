import './App.css';
import React, { Component } from 'react';
import CrsAPP from './components/crs-ui/CrsApp'
import './bootstrap.css';



class App extends Component {
  render() {
    return (
      <div className="App">
        <CrsAPP></CrsAPP>
      </div>
    );
  }
}


export default App;
