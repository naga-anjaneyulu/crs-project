import axios from 'axios';




class CrsService{


     startAssesment(data){

        return axios.post('http://localhost:5000/start', 
        data)

     }

     nextQuestion(data){
      return axios.post('http://localhost:5000/next', 
      data)

   }


   userSatisfaction(data){
      return axios.post('http://localhost:5000/satisfaction', 
      data)
     }
      
     recommendCourses(data){
      return axios.post('http://localhost:5000/recommend', 
      data)
     }

     

     generateReport(data){
      return axios.post('http://localhost:5000/report', 
      data)
     }

}

export default new CrsService()