import './CrsApp.css';
import CrsService from "./crsservice";
import React, {Component} from 'react';



export default class Recommend extends Component{
     constructor(props){
          super(props);
          this.state ={
              recomdata : {},
              count : 0
          }
      }
      
     componentDidMount(){
          const {data} = this.props.location;
          console.log("Mounted");
          console.log(data);
          this.setState({
               recomdata:data
          });

          console.log(this.state.recomdata);
     }


     enableChange = () => {
         
          this.setState({
               count:1
          });

     
     }


     handleChange =(event)=>{
      console.log(event.target.id);
      console.log(event.target.value);
      var data = this.state.recomdata;
      data.gt[event.target.id].choice = event.target.value;
      if(this.state.count > 0){
           this.setState({
               recomdata : data
           });
         
      }
      console.log("after change");
     console.log(this.state.recomdata);
        
     }
  
     report = (event) =>{
   
      var obj ={}
      obj = this.state.recomdata;
      console.log(obj);
        CrsService.generateReport(obj).then(  (response) => {
             console.log('response');
             console.log(response);
             if(response.data.count <= 3){
               this.props.history.push({pathname :"/satisfaction",data: response.data});
             }else{
               this.props.history.push({pathname :"/logout",data: response.data});
             }}).catch((error) => {
                 console.log(error)}); 
  
  }

  render(){ 
     const {data} = this.props.location;  
    
     return(
          <div style ={{ textAlign:"left",  fontFamily:"Verdana, Geneva, sans-serif"}}>
          <HeaderComponent/> 
          <div style ={{fontSize:14,position:"absolute",left:"1370px",fontFamily:"Verdana, Geneva, sans-serif",fontWeight:600,color:"green"}}> Progress : {Math.round(( (2.5 + this.state.recomdata.count) /5.5)*100)}%</div>
          <br></br>
          <h1  style= {{ fontSize :18 ,fontWeight : 600 ,position:'absolute',right:"680px",top : '70px' ,fontFamily:"Verdana, Geneva, sans-serif"}}> User Satisfaction Survey</h1><br></br><br></br>
          <div className = "alert alert-warning" style= {{ fontSize :14 ,fontWeight : 600 , position:'absolute',top : '100px' ,left : "460px",color:"maroon",width:"650px"}}> 
         <p>Please read the instructions carefully :<br></br>
          Let us know , if the following courses would be of any help in obtaining {this.state.recomdata.category} position <br></br>
          Note : Estimated time for this task is 3-5 mins .</p>
          </div><br></br>
          <br></br>
      <div  style ={{overflowX : 'auto',overflowY : 'auto',textAlign:'left',position :'relative' ,top:"60px"}} onClick={this.enableChange}>
      <div>
              
          {  
            Object.keys(this.state.recomdata).length > 0 &&  Object.entries(this.state.recomdata.refine_recommend).map((key,value) => {
               if(key[1].category === this.state.recomdata.category) { return ( <div>
               <div className ="divshadow">
              <p className ="recom2" style ={{fontWeight :650}}>Given the {this.state.recomdata.category}  position , do you think the following course would be helpful in obtaining that position ? </p>
             <p  className ="recom" >Course Name : {key[1].coursename} </p>
             <p className ="recom">Course Description : {key[1].coursedesc}</p>
             <label id ={key[0]} onChange={this.handleChange} className ="radio2">
             <input type="radio" id ={key[0]} value="SA" checked={this.state.recomdata.gt[key[0]].choice === "SA"}  />&nbsp;&nbsp;This course would be extremely helpful in obtaining this position.<br></br>
             <input type="radio"   id ={key[0]} value="A"  checked={this.state.recomdata.gt[key[0]].choice === "A"} />&nbsp;&nbsp;This course would be helpful in obtaining this position. <br></br>
             <input type="radio"  id ={key[0]} value="D"  checked={this.state.recomdata.gt[key[0]].choice === "D"} />&nbsp;&nbsp;This course would be of little help in obtaining  this position.<br></br>
             <input type="radio" id ={key[0]} value="SD"   checked={this.state.recomdata.gt[key[0]].choice === "SD"} />&nbsp;&nbsp;This course is completely irrelevant for this positon<br></br>
             <input type="radio"  id ={key[0]} value="NO"  checked={this.state.recomdata.gt[key[0]].choice === "NO"}/>&nbsp;&nbsp;I'm not really sure whether this course would be of any help.<br></br>
             </label>
             <br></br>
             </div>
             <br></br>
             <br></br>
             </div>
            )} })}
          </div>
          <div>
         
             {  Object.keys(this.state.recomdata).length > 0 && Object.entries(this.state.recomdata.not_recommend).map((key,value) => {
                if(key[1].category === this.state.recomdata.category) { return (  <div>
               <div className ="divshadow">
             <p className ="recom2" style ={{fontWeight :650}} >Given the {this.state.recomdata.category} position , do you think the following course would be helpful in obtaining that position ? </p>
             <p className ="recom" >Course Name : {key[1].coursename} </p>
             <p className ="recom">Course Description : {key[1].coursedesc}</p>
               <label id ={key[0]} onChange={this.handleChange} className = "radio2">
             <input type="radio" id ={key[0]} value="SA" checked={this.state.recomdata.gt[key[0]].choice === "SA"}  />&nbsp;&nbsp;This course would be extremely helpful in obtaining this position.<br></br>
             <input type="radio"   id ={key[0]} value="A"  checked={this.state.recomdata.gt[key[0]].choice === "A"} />&nbsp;&nbsp;This course would be helpful in obtaining this position. <br></br>
             <input type="radio"  id ={key[0]} value="D"  checked={this.state.recomdata.gt[key[0]].choice === "D"} />&nbsp;&nbsp;This course would be of little help in obtaining  this position.<br></br>
             <input type="radio" id ={key[0]} value="SD"   checked={this.state.recomdata.gt[key[0]].choice === "SD"} />&nbsp;&nbsp;This course is completely irrelevant for this positon<br></br>
             <input type="radio"   id ={key[0]} value="NO"  checked={this.state.recomdata.gt[key[0]].choice === "NO"} />&nbsp;&nbsp;I'm not really sure whether this course would be of any help.<br></br>
             </label>
             <br></br>
             </div>
             <br></br>
             <br></br>
             </div>
             )}}) 
              }
          </div>

          <button className ="my-button-3" style={{alignItems:"right"}} onClick={this.report}>Submit</button> 
      </div> 
      </div>   
  )}





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


