import axios from 'axios';

class UserService{


    registerUser(data){

     return axios.post('http://35.225.108.32:5000/register', 
    data)
    }


    loginUser(data){
        return axios.post('http://35.225.108.32:5000/login', 
    data)
    }

}


export default new UserService()