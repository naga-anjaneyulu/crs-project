import axios from 'axios';

class UserService{


    registerUser(data){

     return axios.post('http://35.225.108.32:5000/register', 
    data,{'Access-Control-Allow-Origin': '*','Content-Type':'application/json'})
    }


    loginUser(data){
        return axios.post('http://35.225.108.32:5000/login', 
    data)
    }

}


export default new UserService()