pragma solidity >= 0.8.11 <= 0.8.11;

contract Bitcoin {
    string public bitcoin_data;
    
       
    //call this function to save new user data to Blockchain
    function setData(string memory bd) public {
       bitcoin_data = bd;	
    }
   //get user details
    function getData() public view returns (string memory) {
        return bitcoin_data;
    }

    constructor() public {
        bitcoin_data="";
    }
}