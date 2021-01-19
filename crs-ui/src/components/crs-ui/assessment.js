import './CrsApp.css';
import CrsService from "./crsservice";
import React, {Component} from 'react';

export default class Assessment extends Component{
     
     constructor(props){
         super(props);
         this.state ={
             assdata : {},
             count : 0
         }
     }
     
    
     componentDidMount(){
     console.log("Mounted");
     const {data} = this.props.location;
     console.log(data);
     this.setState({
            assdata:data})
       console.log(this.state.assdata);

    }

    enableChange = () => {
         
     this.setState({
          count:1
     });
}
  
 
 
    handleChange =(event)=>{
     var data = this.state.assdata;
     data.pseudo[event.target.id].response = event.target.value;
      if(this.state.count > 0){
           this.setState({
               assdata : data
           });
      }
      console.log("after change");
     console.log(this.state.assdata);
    }
     
 
    userAssesment = (event) =>{
 
         var obj ={}
         
         obj = this.state.assdata;
         obj.count = this.state.count;
           CrsService.userSatisfaction(obj).then(  (response) => {
                console.log('response');
                console.log(response);
                this.props.history.push({pathname :"/satisfaction",data: response.data});
                   }).catch((error) => {
                    console.log(error) });
    }


  
   
 
    render(){
     const {data} = this.props.location;
     console.log(data);
     console.log(this.state.assdata);
        return(
     <div style={{ textAlign: "left" ,fontFamily:"Verdana, Geneva, sans-serif"}}  onClick={this.enableChange}>
         <HeaderComponent/>
         <div style ={{fontSize : 14,position:"absolute",left:"1370px",fontFamily:"Verdana, Geneva, sans-serif",fontWeight:"600",color:"green"}}> Progress : {Math.round((2/5.5)*100)}%</div>
          <h1  style= {{ fontSize :18 ,fontWeight : 600 ,position:'absolute',right:"700px",top : '60px',fontFamily:"Verdana, Geneva, sans-serif"}}> Self Assessment</h1> 
          <br></br>
          <div className = "alert alert-warning" style= {{ fontSize :14 ,fontWeight : 600 ,position:'absolute',top : '100px' ,left : "460px",color:"maroon",width:"650px"}}> 
         <p>Please read the instructions carefully :<br></br>
          Please find the list of courses , read their corresponding course description carefully and then choose the  appropriate option . <br></br>
          Note : Estimated time for this task is 8-10 mins .</p>
          </div><br></br>
          <br></br>
         <div style ={{position:"relative",top:"120px"}}> 
          {Object.keys(this.state.assdata).length > 0 && Object.entries(this.state.assdata.pseudo).map((key,value) => {
          return(
               <div>
          <div className ="divshadow">
               <p className= "ass" style ={{fontWeight : 650}}>Course Name : &nbsp;{key[1].coursename} </p>
               <p  className= "ass" >Course Description : &nbsp;{key[1].coursedesc} </p>
               <label  className= "ass" >
               <input type="radio" id = {key[0]} value="K" checked={this.state.assdata.pseudo[key[0]].response === "K"} onChange={this.handleChange} />&nbsp;&nbsp;I am familiar with most of the course content completely &nbsp;&nbsp;&nbsp; <br></br>
               <input type="radio" id = {key[0]} value="P" checked={this.state.assdata.pseudo[key[0]].response === "P"} onChange={this.handleChange}/>&nbsp;&nbsp;I am familiar with some of the course content  &nbsp;&nbsp;&nbsp;<br></br>
               <input type="radio" id = {key[0]}  value="DK" checked={this.state.assdata.pseudo[key[0]].response === "DK"} onChange={this.handleChange}/>&nbsp;&nbsp;I do not understand anything,this is something new to me .
             </label>
             <br>
             </br>
          </div>
           <br></br>
           <br></br>
           </div>
          ) 
                }) }  

         <button className ="my-button-3" style={{alignItems:"right"}} onClick={this.userAssesment}>Submit</button>     
         <br></br>
         <br></br> 
       </div>
            
        
         
         {/*<FooterComponent/>*/}
     </div>
 
 
        );
    }
 
 
 
 }





class HeaderComponent extends Component{
  render(){
       return(
             
            <header>
                 <nav className="navbar navbar-expand-md navbar-dark  navbar-dark bg-dark " >
                      <div className ="navbar-brand" style={{fontFamily:"Verdana, Geneva, sans-serif"}}>Course Recommendation System </div>
                      <ul className="navbar-nav navbar-collapse justify-content-end" style={{color:"white"}}>
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


