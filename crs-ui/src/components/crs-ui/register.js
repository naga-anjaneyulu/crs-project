import React, {Component} from 'react'
import { Card, Form, Input, Button } from "antd";
import UserService from "./userservice.js"


class Register extends Component{

   constructor(props){
       super(props);
       this.state = {
           username : '',
           email : '',
           hasRegisterFailed : false
       }
   }


   handleChange = (event) =>{
    this.setState({
        [event.target.name] : event.target.value
    })
   }


   register = (event) =>{

     if(this.state.username.length > 0 && this.state.email.length > 0){
          var obj = { user : { username : this.state.username.toString() , email : this.state.email.toString()  }    }
          UserService.registerUser(obj).then(  (response) => {

            console.log(response);
           var userdata = response.data;
            this.props.history.push({pathname :"/home",state:{user : userdata}});
            
               }).catch((error) => {
                console.log(error) });

     }else{
          this.setState({hasRegisterFailed : true})
     }
    
   }

render(){
     return (
         <div style ={{fontFamily:"Verdana, Geneva, sans-serif"}}>
              <HeaderComponent/>
         <h1  style= {{ fontSize :18 ,fontWeight : 600  ,position:'absolute',right:"600px",top : '70px',fontFamily:"Verdana, Geneva, sans-serif"}}>Course Recommendation System </h1>
         <div className = "alert alert-warning" style= {{ fontSize :16 ,fontWeight : 600  ,position:'absolute',top : '150px' ,left : "440px",color:"maroon"}}>
             <p>Note : Estimated time for completing this task is around 15-20 mins .</p></div><br></br><br></br>
          <ShowInvalidCredentials hasRegisterFailed ={this.state.hasRegisterFailed}> </ShowInvalidCredentials>
         <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '81vh'}}>
         <Card  bordered={false}
         style={{
          border: "1px solid #dcdcdc",
          boxShadow: "0 12px 16px 0 rgba(73, 33, 33, 0.24), 0 17px 50px 0 rgba(80, 28, 28, 0.19)",
          fontFamily: "Verdana, Geneva, sans-serif",
          borderRadius: 10,
          padding :"42px 0px",
          width: 700,
          backgroundColor:"#FFFAFA"
        }}
      >
                   

     Name&nbsp;&nbsp;&nbsp;&nbsp;:&nbsp;&nbsp;&nbsp;&nbsp;<input type ="text" className ="user" name ="username" value ={this.state.username} onChange ={this.handleChange} /><br>
         </br><br></br>
      email&nbsp;&nbsp;&nbsp;&nbsp;:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<input type = "text" className ="user" name="email" value ={this.state.email} onChange ={this.handleChange}/><br>
         </br><br></br>
         <button className ="my-button" style={{alignItems:"center"}} onClick={this.register}>Start</button> <br></br>
        
        
         </Card>
        
         </div>
         
         {/*<FooterComponent/>*/}
         </div>
              );
  }

}



function ShowInvalidCredentials(props){
     if(props.hasRegisterFailed){
         return <p className = "alert alert-warning" style={{color:"red",fontFamily:"Verdana, Geneva, sans-serif" , position:'relative',top : '145px' ,left : "460px",width : "605px"}}>All fields are mandatory !</p>
     }
     return null;
  }
  

class HeaderComponent extends Component{
    render(){
         return(
               
              <header>
                   <nav className="navbar navbar-expand-md navbar-dark bg-dark"  >
                        <div className ="navbar-brand" style={{fontFamily:"Verdana, Geneva, sans-serif"}}> </div>
                   </nav>
              </header>
             
         )
    }
}

class FooterComponent extends Component{
    render(){
         return(
              <footer className="footer" >
                   <span className="text-muted"> All rights reserved 2020 @ Indiana University.</span>
              </footer>
         )
    }
}



export default Register;