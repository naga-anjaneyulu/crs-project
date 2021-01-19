import './CrsApp.css';
import CrsService from "./crsservice";
import React, {Component} from 'react';
import CheckBox from './CheckBox.js'

export default class HomePage extends Component{
    
    constructor(props){
        super(props);
        this.state ={
            user :{ id : 0 },
            categories: [
              {id: 1, value: "Software Developer", isChecked: false},
              {id: 2, value: "Data Scientist", isChecked: false},
              {id: 3, value: "Database Developer", isChecked: false},
              {id: 4, value: "Cyber Security Analyst", isChecked: false}
            ],
            error : false
        }
    }


   componentDidMount(){
     console.log(this.props.location.state.user);
    this.setState({ user:{ id :this.props.location.state.user.id }})}


   startAssesment = (event) =>{
     var obj = {}; var cat_list = [];
     var check = true;
     obj.user = this.props.location.state.user;
     let categories2 = this.state.categories;
     var count = 0;
     categories2.forEach(category => {
          if (category.isChecked === true){
               count += 1;  
          }
       })
       if(count != 3){
          this.setState({error : true})
          check = false;
       }

     if(check){
          categories2.forEach(category => {
               if (category.isChecked === true)
               cat_list.push(category.value) })
            var job = {category1 : cat_list[0],category2:cat_list[1],category3:cat_list[2]}
            obj.job = job;
            //console.log(obj);
            CrsService.startAssesment(obj).then(  (response) => {
               console.log("Response");
               console.log(response);
           this.props.history.push({pathname :"/assessment",data: response.data});
                  }).catch((error) => {
                   console.log(error) });
                  }
   }

   handleCheckChieldElement = (event) => {
     let categories2 = this.state.categories
     categories2.forEach(category => {
        if (category.value === event.target.value)
        category.isChecked =  event.target.checked
     })
     this.setState({categories: categories2})
   }





   render(){
       return(
    <div style ={{fontFamily:"Verdana, Geneva, sans-serif"}}>
        <HeaderComponent/>
        <div style ={{fontSize : 14,position:"absolute",left:"1400px",fontFamily:"Verdana, Geneva, sans-serif",fontWeight:"600",color:"green"}}> Progress : {Math.round((1/5.5)*100)}%</div>
         <h1  style= {{ fontSize :18 ,fontWeight : 600  ,position:'absolute',right:"700px",top : '70px',fontFamily:"Verdana, Geneva, sans-serif"}}> Welcome , {this.props.location.state.user.username}
         </h1> 
         <div className = "alert alert-warning" style= {{ fontSize :14 ,fontWeight : 600 ,position:'absolute',top : '100px' ,left : "460px",color:"maroon",width:"650px"}}> 
         <p>Please read the instructions carefully :<br></br>
          Select 3 career positions you are aspiring for or interested in.</p>
          </div><br></br>
          <ShowInvalidChoice error ={this.state.error}> </ShowInvalidChoice><br></br>
         <div style={{display: 'flex', 
          justifyContent:'center',
           alignItems:'center',
            height: '60vh',
            border: "1px solid #dcdcdc",
            boxShadow: "0 12px 16px 0 rgba(73, 33, 33, 0.24), 0 17px 50px 0 rgba(80, 28, 28, 0.19)",
            fontFamily: "Verdana, Geneva, sans-serif",
            borderRadius: 10,
            backgroundColor:"#FFFAFA",
            top:250,
            width:800,
            position :"absolute",
            left :380

            }}>
        <ul style ={{ fontFamily:"Verdana, Geneva, sans-serif",textAlign :"left" ,position :"absolute",left:"10px",top:"20px"}} >
        { this.state.categories.map((cat, index) => {
        return (<CheckBox     key={index} handleCheckChieldElement={this.handleCheckChieldElement}  {...cat} />)}) }
        </ul>
        <button className ="my-button-2" style={{alignItems:"center"}} onClick={this.startAssesment}>Start Assesment</button>
        </div>
        
        {/*<FooterComponent/>*/}
    </div>

       );
   }
}


function ShowInvalidChoice(props){
     if(props.error){
         return <div className = "alert alert-warning" style={{color:"red",fontFamily:"Verdana, Geneva, sans-serif" , position:'relative',top : '115px' ,left : "460px",width : "650px"}}>Select only 3 categories !</div>
     }
     return null;
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

