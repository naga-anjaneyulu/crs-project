import axios from 'axios';

class UserService{


    registerUser(data){

     return axios.post('http://localhost:5000/register', 
    data)
    }


    loginUser(data){
        return axios.post('http://localhost:5000/login', 
    data)
    }

}


export default new UserService()