import React, {Component} from 'react'
import UserService from "./userservice.js"
import Authentication from "./authentication.js";
import { Card, Form, Input, Button } from "antd";
import {BrowserRouter as Router,Route,Link} from 'react-router-dom';


class Login extends Component{

   constructor(props){
       super(props);
       this.state = {
           username : '',
           password : '',
           hasLoginFailed : false,
           showSuccessMessage : false

       }
   }


   handleChange = (event) =>{
    this.setState({
        [event.target.name] : event.target.value
    })
   }


   login = (event) =>{
    
     var obj = { user : { username : this.state.username.toString() , password : this.state.password.toString()  }    }
          UserService.loginUser(obj).then(  (response) => {
            console.log(response);
            var userdata = response.data;
            Authentication.registerSuccessfulLogin(this.state.username,this.state.password);
            this.props.history.push({pathname :"/home",state:{user : userdata}});
               }).catch((error) => {
                console.log(error) });
                this.setState({hasLoginFailed : true})
   }

render(){
    return (
        <div>
         <HeaderComponent/>
        <h1 style={{ textAlign: "center" }}>Login</h1>
       
        <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '81vh'}}>
        <Card  bordered={false}
        style={{
         border: "1px solid #dcdcdc",
         boxShadow: "0 12px 16px 0 rgba(73, 33, 33, 0.24), 0 17px 50px 0 rgba(80, 28, 28, 0.19)",
         fontFamily: "Verdana, Geneva, sans-serif",
         borderRadius: 10,
         padding :"32px 0px",
         width: 700,
         backgroundColor:"#FFFAFA"
       }}
     >
    Username &nbsp;&nbsp;&nbsp;&nbsp;:&nbsp;&nbsp;&nbsp;&nbsp; <input type ="text" className ="user" name ="username" value ={this.state.username} onChange ={this.handleChange} /><br>
        </br><br></br>
Password &nbsp;&nbsp;&nbsp;&nbsp;:&nbsp;&nbsp;&nbsp;&nbsp;  <input type = "password" className ="user" name="password" value ={this.state.password} onChange ={this.handleChange}/><br>
        </br><br></br><br></br>
        <button className ="my-button" onClick={this.login}>Login</button> <br></br>
        <ShowInvalidCredentials hasLoginFailed ={this.state.hasLoginFailed}> </ShowInvalidCredentials>
       
        </Card>
        </div>
        {/*<FooterComponent/>*/}
        </div>
             );
 }

}


function ShowInvalidCredentials(props){
   if(props.hasLoginFailed){
       return <div className = "alert alert-warning" style={{color:"red"}}>Invalid Credentials !</div>
   }
   return null;
}


class HeaderComponent extends Component{
    render(){
         return(
               
              <header>
                   <nav className="navbar navbar-expand-md navbar-dark bg-dark" >
                        <div className ="navbar-brand" style={{fontFamily:"Verdana, Geneva, sans-serif"}}>Course Recommendation System </div>
                        <ul className="navbar-nav navbar-collapse justify-content-end" style={{color:"white"}}>
                             <li><Link className= "nav-link"  to="/loginPage">Login</Link></li>&nbsp;&nbsp;&nbsp;
                             <li><Link className= "nav-link"  to="/registerPage">Register</Link></li>
                        </ul>
                   </nav>
              </header>
             
         )
    }
}

class FooterComponent extends Component{
    render(){
         return(
              <footer className="footer"  >
                   <span className="text-muted"> All rights reserved 2020 @ Indiana University.</span>
              </footer>
         )
    }
}



export default Login;