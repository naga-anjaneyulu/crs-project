import './CrsApp.css';
import CrsService from "./crsservice";
import React, {Component} from 'react';



export default class Satisfaction extends Component{
   
    constructor(props){
        super(props);
        this.state ={
            satdata : {},
            refine :"Good"
            
        }
    }
    
   
   componentDidMount(){
     console.log("Mounted");
      const {data} = this.props.location;
    this.setState({
           satdata:data
           
      })
      console.log(this.state.satdata);

   }



     

   handleChange =(event)=>{
       console.log("handle change");
       console.log(event.target.value);
    this.setState({
        refine:event.target.value
    });
   }
   handleChange2 =(event)=>{
    this.setState({
        qa:event.target.value
    });
   }

   recommend = (event) =>{
 
    var obj ={}
    obj = this.state.satdata;
    obj.refine_quality = this.state.refine;
    console.log(obj);
      CrsService.recommendCourses(obj).then(  (response) => {
           console.log('response');
           console.log(response);
           this.props.history.push({pathname :"/recommend",data:response.data});
              }).catch((error) => {
               console.log(error) });

    
  
      
  

}


   render(){
    const {data} = this.props.location;
    var our_count = 0;
    var raw_count = 0;
    return(
            <div style={{ textAlign: "left" ,fontFamily:"Verdana, Geneva, sans-serif"}}>
            <HeaderComponent/>
            <div style ={{fonstSize :14,position:"absolute",left:"1370px",fontFamily:"Verdana, Geneva, sans-serif",fontWeight:600,color:"green"}}> Progress : {Math.round(( (2 + this.state.satdata.count) /5.5)*100)}%</div>
             <br></br>
            <h1  style= {{ fontSize :18 , fontWeight : 600 ,position:'absolute',right:"700px",top : '70px',fontFamily:"Verdana, Geneva, sans-serif" }}> User Satisfaction Survey</h1><br></br><br></br>
            <div className = "alert alert-warning" style= {{ fontSize :14 ,fontWeight : 600 , position:'absolute',top : '100px' ,left : "400px",color:"maroon",width:"650px"}}> Given the {this.state.satdata.category} position , read the following recommended course lists and answer the question below . </div><br></br><br></br>
          <div style ={{ display:'flex',flexDirection:'row',justifyContent: 'space-around',position :'relative' ,top:"10px"}}>
            <div className ="sat-list">
            <p className = "sat-title"> Recommendation List 1 <br></br></p>
            <ul >
             { Object.keys(this.state.satdata).length > 0 && Object.entries(this.state.satdata.refine_recommend).map((key,value) => {
                  if( key[1].category === this.state.satdata.category){
                    our_count = our_count + 1; 
                   return  ( 
                    <li className="dd-list-item" >{key[1].coursename}<br></br></li>
                      )
               }

                 })}


            </ul>    
            </div> 
            <div  className ="sat-list" >
                 <p className = "sat-title"> Recommendation List 2 <br></br></p>
            <ul >          
             { 
                 Object.keys(this.state.satdata).length > 0 && Object.entries(this.state.satdata.pseudo_recommend).map((key,value) => {
                                  if(our_count < 5 && raw_count <5 && key[1].category === this.state.satdata.category  ){
                                   raw_count = raw_count + 1; 
                                       return    ( <li className="dd-list-item" >{key[1].coursename}<br></br> </li>)
                                       
                                  }else if ( our_count >=  5 && our_count > raw_count && key[1].category === this.state.satdata.category){;
                                   raw_count = raw_count + 1; 
                                   return    ( <li className="dd-list-item" >{key[1].coursename} <br></br></li>)
                                  }
          
             })    }
            
            </ul>    
            </div>
          </div>
            <br></br>
            <br></br>
          <div className ="question">
            <p className ="ques">Which one do you think is appropriate for {this.state.satdata.category} career. </p>
              <label className = "ques">
             <input type="radio" value="Good" checked={this.state.refine === "Good"} onChange={this.handleChange} />&nbsp;&nbsp;List 1 is appropriate for {  this.state.satdata.category } position .<br></br>
             <input type="radio"   value="Bad" checked={this.state.refine === "Bad"} onChange={this.handleChange}/>&nbsp;&nbsp;List 2 is appropriate for {  this.state.satdata.category } position .<br></br>
             <input type="radio"   value="Similar" checked={this.state.refine === "Similar"} onChange={this.handleChange}/>&nbsp;&nbsp;They are similar .
             </label>
          </div>


         <button className ="my-button-4" style={{alignItems:"right"}} onClick={this.recommend}>Submit</button> 
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


