import axios from 'axios';




class CrsService{


     startAssesment(data){

        return axios.post('http://'+window.location.host+':5010/start', 
        data)

     }

     nextQuestion(data){
      return axios.post('http://'+window.location.host+':5010/next', 
      data)

   }


   userSatisfaction(data){
      return axios.post('http://'+window.location.host+':5010/satisfaction', 
      data)
     }
      
     recommendCourses(data){
      return axios.post('http://'+window.location.host+':5010/recommend', 
      data)
     }

     

     generateReport(data){
      return axios.post('http://'+location.hostname+':5010/report', 
      data)
     }

}

export default new CrsService()