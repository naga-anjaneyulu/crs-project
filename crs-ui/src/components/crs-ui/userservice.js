import axios from 'axios';

class UserService{


    registerUser(data){

     return axios.post('http://localhost:5000/register', 
    data,{'Access-Control-Allow-Origin': '*','Content-Type':'application/json',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, PATCH, DELETE',
    "Access-Control-Allow-Headers": 'x-access-token, Origin, X-Requested-With, Content-Type, Accept'})
    }


    loginUser(data){
        return axios.post('http://35.225.108.32:5000/login', 
    data)
    }

}


export default new UserService()