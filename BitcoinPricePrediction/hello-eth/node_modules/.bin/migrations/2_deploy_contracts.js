const Bitcoin = artifacts.require("Bitcoin");

module.exports = function(deployer) {
  deployer.deploy(Bitcoin);
};
