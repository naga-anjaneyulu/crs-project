class Authentication{

      registerSuccessfulLogin(username,password){
          sessionStorage.setItem('authenticatedUser',username);
      }

      logout(){
          sessionStorage.removeItem('authenticatedUser');
      }

}


export default new Authentication()