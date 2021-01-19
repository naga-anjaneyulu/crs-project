import './CrsApp.css';
import Login from './login';
import Register from './register';
import HomePage from './homePage';
import Satisfaction from './satisfaction';
import Assessment from './assessment';
import Recommend from './recommend';
import React, {Component} from 'react';
import Authentication from "./authentication.js";
import { render } from '@testing-library/react';
import {BrowserRouter as Router,Route,Switch,Link} from 'react-router-dom';
class CrsApp extends Component{

render(){
     return (
          <div className = "CrsApp" >
             <Router>
                 
             <Switch> 
             <Route exact path="/"  component={Register}/>
             <Route path="/registerPage" component={Register}/>
             <Route path="/home" component={HomePage}/>
             <Route path= "/assessment" component={Assessment}/>
             <Route path ="/satisfaction" component ={Satisfaction}/>
             <Route path= "/recommend" component={Recommend}/>
             <Route path= "/logout" component={Logout}/>
             <Route component ={ErrorComponent}/>
             </Switch>
            
             </Router>
          </div>

     );
}

}


class Logout extends Component{
     render(){
          return(
               <div>
                    <h1 style= {{ fontSize :28 ,fontWeight : 600 ,position:'absolute',right:"700px",top : '200px',fontFamily:"Verdana, Geneva, sans-serif"}}>Thank you !</h1>
                    
               </div>
          )
     }
}


function ErrorComponent(){
     
     return  <div > Oop's !! Something went wrong, seek help !  </div>
     
}



export default CrsApp;