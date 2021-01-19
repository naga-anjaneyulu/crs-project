import axios from 'axios';

class UserService{


    registerUser(data){

     return axios.post('http://'+window.location.host+':5010/register', 
    data)
    }


    loginUser(data){
        return axios.post('http://'+window.location.host+':5010/login', 
    data)
    }

}


export default new UserService()